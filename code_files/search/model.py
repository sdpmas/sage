# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class SearchModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args=None,searcher=None,codebase=None):
        super(SearchModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.searcher=searcher 
        self.codebase=codebase
    def get_representation_batch(self,qc_ids=None,device=None):
        """get represenations in batch for either queries or codes"""
        return self.encoder(qc_ids.to(device),attention_mask=qc_ids.ne(1).to(device))[1]

    def get_representation_one(self,query,device=None):
        """get representation for a single query: less dataset stuffs."""
        query_tokens=[self.tokenizer.cls_token]+self.tokenizer.tokenize(query)[:298]+[self.tokenizer.sep_token]
        query_ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(query_tokens)).unsqueeze(dim=0).to(device)
        return self.encoder(query_ids,attention_mask=query_ids.ne(1))[1].squeeze(dim=0)

    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        # print('original shapes: ',code_inputs.shape,nl_inputs.shape)
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        # print(f'outputs:{outputs.shape}')
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        # print(f'inter: {code_vec.shape,nl_vec.shape}')
        
        if return_vec:
            return code_vec,nl_vec
        #bs*1*length_nl*dim * 1*bs*length_co*dim 
        # print(f'before scores: {nl_vec[:,None,:].shape,code_vec[None,:,:].shape}')
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        # print('before sum: ',(nl_vec[:,None,:]*code_vec[None,:,:]).shape)
        # print('scores: ',scores.shape)
        # print('arange: ',torch.arange(bs).shape)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        # exit()
        return loss,code_vec,nl_vec

      
        
 
