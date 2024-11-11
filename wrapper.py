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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from models import (
    InputEncoder,
    PromptEncoder,
)

import logging

logger = logging.getLogger(__name__)




class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 max_seq_length: int,
                 label_list: List[str],
                 pattern_id: int = 0,  # indicate the length of the continuous prompt tokens
                 cache_dir: str = None,
                 output_dir=None,
                 embed_size=128,
                 prompt_encoder_type="lstm", # ["lstm", "sparse-attention"]
                 eval_every_step=20):

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.embed_size = embed_size
        self.prompt_encoder_type = prompt_encoder_type
        self.eval_every_step = eval_every_step
        
        




class ModelWrapper(nn.Module):
    def __init__(self, config: WrapperConfig):
        self.config = config
        
        
    
    
    
    
    def from_pretrained(self,):
        pass
    
    
    
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