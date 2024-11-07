import os
import torch
import evaluate
import numpy as np
from config import Config


from torch.utils.data import DataLoader

from datasets import (
    Dataset,
    load_dataset
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
)



from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support



def get_model_name_using_model(model):
    config = model.config  
    # 尝试直接获取模型的名称  
    if hasattr(config, 'name_or_path') and config.name_or_path is not None:  
        # 使用 os.path.basename 提取路径中的模型名称  
        model_name = os.path.basename(config.name_or_path)  
        return model_name  
    # 根据模型类型和隐藏层大小推断模型名称  
    if config.model_type == "bert":  
        if config.hidden_size == 768:  
            return "bert-base-uncased"  
        elif config.hidden_size == 1024:  
            return "bert-large-uncased"  
    elif config.model_type == "roberta":  
        if config.hidden_size == 768:  
            return "roberta-base"  
        elif config.hidden_size == 1024:  
            return "roberta-large"  
    elif config.model_type == "llama":  
        if config.hidden_size == 4096:  
            return "meta-llama/Llama-2-13b-hf"  
        elif config.hidden_size == 5120:  
            return "meta-llama/Llama-2-70b-hf"  
    elif config.model_type == "qwen2":  
        if config.hidden_size == 4096:  
            return "qwen-14b-chat"  
        elif config.hidden_size == 5120:  
            return "qwen-16b-chat"  
    else:  
        # 无法匹配已知模型，返回未知模型提示  
        return "unknown model, please check your config, it should be [bert | llama | qwen2]"  


def get_vocab_embeddings_from_model(model, token_ids:torch.LongTensor):
    '''
     model is a pretrained model
     
     token_ids: a tensor of shape (num_tokens, 1)
     
     return a tensor of shape (num_tokens, 1, hidden_size)
    '''
    if hasattr(model, 'embeddings'):  
        return model.embeddings(token_ids)  
    elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):  
        return model.bert.embeddings(token_ids)  
    elif hasattr(model, 'roberta') and hasattr(model.roberta, 'embeddings'):  
        return model.roberta.embeddings(token_ids)  
    elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'embeddings'):  
        return model.distilbert.embeddings(token_ids)   
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'embeddings'):  
        return model.base_model.embeddings(token_ids)  
    else:  
        raise AttributeError(f"Can not find the embedding layer in the model. Please check the model type {type(model).__name__}.") 




def prepare_model_tokenizer(model_path, auto_model_class = AutoModel, tokenizer_path = None):
    '''
     return model, tokenizer
    '''
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, num_labels=4)
    
    model = auto_model_class.from_pretrained(model_path, config=config)
    
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    print(f"Model's current num_labels: {model.config.num_labels}") 
     
    model_config = model.config
    model_name_or_path = model_config.name_or_path
    print("model_name_or_path = ", model_name_or_path)
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    print("padding_side = ", padding_side)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    
    return model, tokenizer
