import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
    
class SearchGenModel(nn.Module):   
	def __init__(self, search_model,py_gen_model,gen_tokenizer,js_gen_model):
		super(SearchGenModel, self).__init__()
		self.search_model=search_model 
		self.py_gen_model=py_gen_model 
		self.gen_tokenizer=gen_tokenizer
		self.js_gen_model=js_gen_model 