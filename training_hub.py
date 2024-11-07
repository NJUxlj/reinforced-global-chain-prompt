'''

this file is used train all the models using all the peft methods



'''
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

from utils import (
    prepare_model_tokenizer, 
)


from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


from p_tuning_v2 import train_p_tuning_v2
from p_tuning import train_p_tuning
from prompt_tuning import train_prompt_tuning
from prefix_tuning import train_prefix_tuning
from lora import train_lora

peft_dict = {  
    "p_tuning": train_p_tuning,
    "p_tuning_v2": train_p_tuning_v2,
    "prompt_tuning": train_prompt_tuning,
    "prefix_tuning": train_prefix_tuning,
    "lora": train_lora
}


 


def train_all():
    
    model_names = ["bert-base-uncased"]
    model_info_dict = Config['models']
    
    for model_name, model_info in model_info_dict.items():
        model_path = model_info['model_path']
        
        if model_name in model_names:
            model, tokenizer = prepare_model_tokenizer(model_path)
            for method in peft_dict.keys():
                print(f"========== Training {model_name} with {method} ==============")
                peft_dict[method](model, tokenizer)






def evaluate_all():
    model_names = ["bert-base-uncased", "bert-large-uncased"]
    
    
    
    


if __name__ == '__main__':
    train_all()
