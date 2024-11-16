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


from p_tuning_v2 import *
from p_tuning import *
from prompt_tuning import *
from prefix_tuning import *
from lora import *
from bidirectional_prompt_tuning2 import *

peft_dict = {  
    "p_tuning": train_p_tuning,
    "p_tuning_v2": train_p_tuning_v2,
    "prompt_tuning": train_prompt_tuning,
    "prefix_tuning": train_prefix_tuning,
    "lora": train_lora,
    "bidirectional_prompt_tuning": train_bidirectional_prompt_tuning,
}


 


def train_all():
    
    model_names = ["bert-base-uncased"]
    # pt_names = ["p_tuning_v2", "prompt_tuning", "prefix_tuning", "p_tuning"]
    pt_names = ["prompt_tuning", "p_tuning"]
    
    # dataset_names = ['race', 'sciq', 'dream', 'commonsense_qa']
    dataset_names = ['race']
    
    
    
    model_info_dict = Config['models']
    
    for model_name, model_info in model_info_dict.items():
        model_path = model_info['model_path']
        
        if model_name in model_names:
            # model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)
            for method in peft_dict.keys():
                if method in pt_names:
                    # print(f"========== Training {model_name} with {method} on dataset: {dataset_names} ==============")
                    print()
                    for dataset_name in dataset_names:
                        print(f"==========Training {model_name} with {method} on dataset: {dataset_name}")
                        
                        if method == "p_tuning_v2":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels = 4)
                            
                            max_length = get_max_length_from_model(model)
                            config = PtuningV2Config(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name = dataset_name,
                                max_seq_length = max_length,
                            )
                            peft_dict[method](config)

                        elif method == "p_tuning":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels = 4)
                            
                            max_length = get_max_length_from_model(model)
                            config = PtuningConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name = dataset_name,
                                max_seq_length = max_length,
                            )
                            peft_dict[method](config)
                        
                        elif method == "prefix_tuning":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels = 4)
                            
                            max_length = get_max_length_from_model(model)
                            config = PrefixTuningTrainerConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name = dataset_name,
                                max_seq_length = max_length,
                            )
                            peft_dict[method](config)
                            
                        elif method == "prompt_tuning":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels = 4)
                            
                            max_length = get_max_length_from_model(model)
                            config = PromptTuningTrainerConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name = dataset_name,
                                max_seq_length = max_length,
                            )
                            peft_dict[method](config)
                            
                        else:
                            raise ValueError(f"no such peft method: {method}")

def evaluate_all():
    model_names = ["bert-base-uncased", "bert-large-uncased"]
    
    
    
    


if __name__ == '__main__':
    train_all()
