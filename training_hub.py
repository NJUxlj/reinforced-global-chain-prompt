'''

this file is used train all the models using all the peft methods



'''
import os
import torch
import evaluate
import numpy as np
from config.config import Config


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
from models.baas_prompt import *

peft_dict = {  
    "p_tuning": train_p_tuning,
    "p_tuning_v2": train_p_tuning_v2,
    "prompt_tuning": train_prompt_tuning,
    "prefix_tuning": train_prefix_tuning,
    "lora": train_lora,
    "baas_prompt": train_baas_prompt,
}


 


def train_all():
    
    model_names = ["bert-base-uncased"]
    # pt_names = ["p_tuning_v2", "prompt_tuning", "prefix_tuning", "p_tuning"]
    pt_names = ["prompt_tuning", "baas_prompt"]
    
    dataset_names = ['race', 'sciq', 'dream', 'commonsense_qa']
    # dataset_names = ['race']
    
    
    
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
                            model, tokenizer = prepare_model_tokenizer(model_path,AutoModelForSequenceClassification, model_path)

                            max_seq_length = get_max_length_from_model(model)
                            hidden_size = get_hidden_size_using_model(model)

                            config = PtuningV2Config(
                                model_name = model_name,
                                model_path=model_path,
                                auto_model_class = AutoModelForSequenceClassification,
                                dataset_name=dataset_name,
                                # learning_rate=Config['learning_rate'],
                                max_seq_length=max_seq_length,
                                num_labels=2,
                                # batch_size=Config['batch_size'],
                                # num_epochs = Config['num_epochs'],
                                prefix_projection=True,
                                prefix_hidden_size=hidden_size,
                                prefix_length=100
                            )
                            
                            peft_dict[method](config)

                        elif method == "p_tuning":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)

                            max_seq_length = get_max_length_from_model(model)
                            hidden_size = get_hidden_size_using_model(model)
                            
                            config = PtuningConfig(
                                model_name = model_name,
                                model_path = model_path,
                                max_seq_length=max_seq_length,
                                dataset_name= dataset_name,
                                num_epochs=5,
                                num_labels=2,
                                prefix_hidden_size=hidden_size,
                                encoder_hidden_size=hidden_size,
                                prefix_length=100,
                                
                            )
                            
                            peft_dict[method](config)
                        
                        elif method == "prefix_tuning":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels=2)
    
                            model_name = get_model_name_using_model(model)
                            dataset_name = 'race'
                            
                            max_seq_length = get_max_length_from_model(model)
                            
                            hidden_size = get_hidden_size_using_model(model)
                            
                            config = PrefixTuningTrainerConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name=dataset_name,
                                max_seq_length= max_seq_length,
                                num_epochs=5,
                                num_labels=2,
                                prefix_length=10,
                                prefix_hidden_size=hidden_size,
                                prefix_projection_hidden_size=4*hidden_size,
                                encoder_hidden_size=hidden_size,
                                
                            )
                            peft_dict[method](config)
                            
                        elif method == "prompt_tuning":
                            if dataset_name=='race':
                                continue
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels=2)

                            max_seq_length = get_max_length_from_model(model)
                            hidden_size = get_hidden_size_using_model(model)

                            config = PromptTuningTrainerConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name= dataset_name,
                                max_seq_length=max_seq_length,
                                num_epochs=5,
                                num_labels=2,
                                prefix_hidden_size=hidden_size,
                                encoder_hidden_size=hidden_size,
                                batch_size=4
                            )
                            peft_dict[method](config)
                        
                        elif method == "baas_prompt":
                            model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification)
    
                            max_seq_length = get_max_length_from_model(model)
                            import math
                            prefix_length = 10
                            suffix_length = math.floor(0.1 * max_seq_length)
                            
                            hidden_size = get_hidden_size_using_model(model)

                            config = BaasPromptConfig(
                                model_name = model_name,
                                model_path = model_path,
                                dataset_name=dataset_name,
                                max_seq_length=max_seq_length,
                                num_epochs=5,
                                num_labels=2,
                                all_layers=False,
                                is_prefix=False,
                                prefix_projection=True,
                                prefix_hidden_size=hidden_size,
                                encoder_hidden_size=hidden_size,
                                prefix_length=prefix_length,
                                suffix_length=suffix_length,
                                batch_size=4,
                                
                            )
                            
                            args=ChainEncodingArguments(
                                dataset=config.dataset_name,
                                hidden_size=config.prefix_hidden_size, # 这个hidden_size最终会传给encode cot chain用到的 sentence transformer
                                output_dir=f"./autocot/experiment/{dataset_name}",
                                embedding_dir = f"./autocot/embeddings/{dataset_name}",
                                context_dir = f"./autocot/context/{dataset_name}",
                                temperature=0.7
                            )
                            # train_baas_prompt(config, args)
                            peft_dict[method](config,args)
                            
                        else:
                            raise ValueError(f"no such peft method: {method}")

def evaluate_all():
    model_names = ["bert-base-uncased", "bert-large-uncased"]
    
    
    
    


if __name__ == '__main__':
    train_all()
