import torch
import torch.nn as nn
import torch.nn.functional as F  

from transformers import AutoModel, AutoTokenizer
from config import Config   

from typing import List, Tuple, Dict, Optional
import math
import json
import jsonpickle
import os

import numpy as np
from sklearn.metrics import f1_score

from tqdm import trange, tqdm
from transformers.data.metrics import simple_accuracy

from transformers import (
    AutoConfig,
    BertConfig,
    Qwen2Config,
    RobertaConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    Qwen2Tokenizer,
    RobertaTokenizer,
    
    BertForSequenceClassification,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    RobertaForSequenceClassification,
    set_seed,
)

from peft import (
    PeftModel
)

from .models import (
    InputEncoder,
)

from config import *

import logging

logger = logging.getLogger(__name__)



# MLM_WRAPPER = "mlm"
# CLM_WRAPPER = "clm"

SEQ_CLS_WRAPPER = "seq-cls"

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQ_CLS_WRAPPER: BertForSequenceClassification
    },
    'qwen2': {
        'config': Qwen2Config,
        'tokenizer': Qwen2Tokenizer,
        SEQ_CLS_WRAPPER: Qwen2ForSequenceClassification
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQ_CLS_WRAPPER: RobertaForSequenceClassification
    }
}





class PromptEncoder(nn.Module):
    
    '''
    function:
        1.使用 SparseAttention来编码MCQ多项选择数据集的input
        
    
    '''
    def __init__(self, config:WrapperConfig, tokenizer): 
        super(PromptEncoder, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embedding_size = config.embedding_size
        self.prompt_length = self.config.pattern_id
        
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels = len(config.label_list),
            finetuning_task = config.task_name,
            cache_dir = config.cache_dir if config.cache_dir else None,
            use_cache = False
        )
        
        
        model_class = MODEL_CLASSES[self.config.model_type][SEQ_CLS_WRAPPER]



class ModelWrapper(nn.Module):
    def __init__(self, config: WrapperConfig):
        self.config = config
        
        
        
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        
        self.tokenizer = tokenizer_class.from_pretrained(
            
        )
        
        
        self.model = PromptEncoder(config, self.tokenizer)
        
    
    
    
    
    def from_pretrained(self,)->'ModelWrapper':
        """Load a pretrained wrapper from a given path."""

        
    
    
    def save(self, path: str) -> None:
        logger.info("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict()
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict()
            }
        elif self.config.prompt_encoder_type == "sparse-attention":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "sparse_attention": model_to_save.sparse_attention.state_dict()
            }
        else:
            raise ValueError("unknown prompt_encoder_type.")

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)
    
    
    
    def train(self,):
        pass