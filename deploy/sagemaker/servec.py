
import torch
import os
import logging
import sys
import json
import pickle 

import io,tokenize
import sys
sys.path.append(".")
import scann 
# from annoy import AnnoyIndex
from code_files.search.model import SearchModel
from code_files.gen.seargen import SearchGenModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from code_files import config
import transformers
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("model")

def model_fn(model_dir):
    print('we get here')
    log.info('got model_fn')
    log.info("In model_fn(). DEPLOY_ENV="+str(os.environ.get("DEPLOY_ENV")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_config_rob = RobertaConfig.from_pretrained("microsoft/codebert-base")
    search_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    search_model = RobertaModel.from_pretrained("microsoft/codebert-base",config=search_config_rob,cache_dir=None) 
    searcher = scann.scann_ops_pybind.load_searcher(os.path.join(config.DATA_DIR,'scann_searcher/'))
    # f=768
    # indices=AnnoyIndex(f,'angular')
    # indices.load('../search/data/indices.ann')
    # codebase=pickle.load(open('../search/data/codebase.bin','rb'))
    print('checkpoint222')
    log.info('checkpoint222')
    
    codebase=pickle.load(open(os.path.join(config.DATA_DIR,'codebase.bin'),'rb'))
    search_model=SearchModel(encoder=search_model,config=search_config_rob,tokenizer=search_tokenizer,args=None,codebase=codebase,searcher=searcher)

    print('loading gen model')
    gen_tokenizer=transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    gen_model=transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    seargen_model=SearchGenModel(search_model=search_model,gen_model=gen_model,gen_tokenizer=gen_tokenizer)
    seargen_model.load_state_dict(torch.load(os.path.join(model_dir,'model.pth')))
    print('checkpoint3')
    log.info('checkpoint3')
    seargen_model.to(device).eval()
    return seargen_model 
    

# From docs:
# Default json deserialization requires request_body contain a single json list.
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py#L49
def input_fn(request_body, request_content_type):
    log.info('Deserializing the input data.')
    input_data = json.loads(request_body)
#     query=input_data['query']
    return input_data 
    

def predict_fn(input_data, model):
    def remove_comments_and_docstrings(source):
        io_obj = io.StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        out = '\n'.join(l for l in out.splitlines() if l.strip())
        return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info('Generating prediction based on input parameters.')

    with torch.no_grad():
        query_repr=model.search_model.get_representation_one('NL: '+input_data['query'],device=device)
    # codes=model.indices.get_nns_by_vector(query_repr,3)
#     ex_code_idx=[]
    ex_code_idx,_=model.search_model.searcher.search(query_repr.detach().cpu().numpy(), final_num_neighbors=5)
    # ex_code_idx=model.search_model.searcher.get_nns_by_vector(query_repr,5)
    ex_codes=[{'code':model.search_model.codebase[c]['code'],'url':model.search_model.codebase[c]['url']} for c in ex_code_idx]
    filtered_ex_codes=[]
    query_prompt="Query:\n"+input_data['query']+'\n'
    # context_prompt='Context:\n'+'None\n'
    excode_prompt='Examples from search:\n'
    if ex_codes:
        for i_e_code,e_code in enumerate(ex_codes):
            if i_e_code>1:
                filtered_ex_codes.append({'code':e_code['code'],'url':e_code['url']})
                continue

            try:
                f_e_code=remove_comments_and_docstrings(e_code['code'])
                filtered_ex_codes.append({'code':f_e_code,'url':e_code['url']})
                excode_prompt+=f_e_code
                excode_prompt+='\n'
            except:
                if i_e_code==1:
                    excode_prompt+='\n'
                continue
    else:
        excode_prompt+='None\n'
    max_context_len=200
    query_ids=model.gen_tokenizer.encode(query_prompt,verbose=False)
    if input_data['context']:
        context_ids=model.gen_tokenizer.encode('Context:\n')+model.gen_tokenizer.encode(input_data['context']+'\n')[-max_context_len:]
    else:  
        context_ids=model.gen_tokenizer.encode('Context:\n')+model.gen_tokenizer.encode('None\n')

    excode_ids=model.gen_tokenizer.encode(excode_prompt,verbose=False)
    total_query_ids=context_ids+query_ids+excode_ids
    add_newline=False 
    #dynamically add block size
    if len(total_query_ids)>450:
        add_newline=True
    total_query_ids=total_query_ids[:450]
    if not add_newline:
        input_ids=torch.LongTensor(total_query_ids+model.gen_tokenizer.encode('Generate Code:\n',verbose=False)).unsqueeze(dim=0).to(device)	
    else:
        input_ids=torch.LongTensor(total_query_ids+model.gen_tokenizer.encode('\nGenerate Code:\n',verbose=False)).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        output_ids=model.gen_model.generate(
                        input_ids,
                        num_beams=2,
                        early_stopping=True,
                        max_length=700,num_return_sequences=2
                    ).squeeze(dim=0)
    decoded_outs=[]
    for o_id in output_ids:
        decoded_outs.append(model.gen_tokenizer.decode(o_id).split("Generate Code:\n")[1].replace("<|endoftext|>", ""))
    return_dict={'code':decoded_outs,'examples':filtered_ex_codes}
    return return_dict


def output_fn(prediction, content_type):
    return json.dumps(prediction)

# if __name__=='__main__':
    
#     model_dir='.'
#     model=model_fn(model_dir=model_dir)
#     request_body=json.dumps({'query':'concatenate two dicts'})
#     query=input_fn(request_body=request_body,request_content_type=None)
#     prediction=predict_fn(input_data=query,model=model)
#     pred=output_fn(prediction=prediction,content_type=None)
#     results=json.loads(pred)
#     for c in results['examples']:
#         print(c)
#         print('------------')
#     print('***********************')
#     print(results['code'])
#     print('***********************')


    