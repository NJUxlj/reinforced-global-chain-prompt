import torch
import platform
import multiprocessing  
import psutil  
import json
import os 

from abc import ABC

from typing import List, Tuple, Dict, Optional

system_name = platform.system() 

# AutoDL: /root/autodl-tmp/models/
# aliyun-dsw: "/mnt/data/models/"   

MODEL_BASE_PATH = ""
if system_name == "Windows":
    MODEL_BASE_PATH = "D:\\pre-trained-models\\"
elif system_name == "Linux":
    MODEL_BASE_PATH = "/root/autodl-tmp/models/"   

print("MODEL_BASE_PATH: ", MODEL_BASE_PATH)

# 所有模型的上级目录
# MODEL_BASE_PATH = "\\mnt\\data\\models"

NUM_PROCESSES = 0

if torch.cuda.is_available():  
        NUM_PROCESSES = torch.cuda.device_count()  
        print(f"检测到 {NUM_PROCESSES} 个可用的 GPU。")  
else:  
    print("未检测到 GPU，将使用 CPU 进行训练。")  


MODEL_WEIGHT_NAME = "/model.pt"

Config = {
    "output_dir":"./output",
    
    "vocab":"data/chars.txt",
    "train_data_path": "data/train.csv",
    "test_data_path": "data/test.csv",
    "submission_path": "submission.csv",
    

    "seed": 42,
    "num_layers": 4,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 2,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model.pth",
    
    
    "save_model_dir":{
        "bert-base-uncased": {
            "prompt-tuning":{
                "race": "save/bert-base-uncased/prompt-tuning/race"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/prompt-tuning/race-m"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/prompt-tuning/race-h"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/prompt-tuning/race-c"+MODEL_WEIGHT_NAME,
                "record": "save/bert-base-uncased/prompt-tuning/record"+MODEL_WEIGHT_NAME, # ReCoRD
                "multirc": "save/bert-base-uncased/prompt-tuning/multirc"+MODEL_WEIGHT_NAME,  #MultiRC
                "arc": "save/bert-base-uncased/prompt-tuning/arc"+MODEL_WEIGHT_NAME,   #ARC
                "dream": "save/bert-base-uncased/prompt-tuning/dream"+MODEL_WEIGHT_NAME, # DREAM
                "sciq": "save/bert-base-uncased/prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
            },
            "prefix-tuning":{
                "race": "save/bert-base-uncased/prefix-tuning/race"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/prefix-tuning/race-m"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/prefix-tuning/race-h"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/prefix-tuning/race-c"+MODEL_WEIGHT_NAME,
                "record": "save/bert-base-uncased/prefix-tuning/record"+MODEL_WEIGHT_NAME,
                "multirc": "save/bert-base-uncased/prefix-tuning/multirc"+MODEL_WEIGHT_NAME,
                "arc": "save/bert-base-uncased/prefix-tuning/arc"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-base-uncased/prefix-tuning/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-base-uncased/prefix-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/prefix-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,

            },
            "bidirectional-prompt-tuning":{
                "race": "save/bert-base-uncased/bidirectional-prompt-tuning/race"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/bidirectional-prompt-tuning/race-m"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/bidirectional-prompt-tuning/race-h"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/bidirectional-prompt-tuning/race-c"+MODEL_WEIGHT_NAME,
                "record": "save/bert-base-uncased/bidirectional-prompt-tuning/record"+MODEL_WEIGHT_NAME,
                "multirc": "save/bert-base-uncased/bidirectional-prompt-tuning/multirc"+MODEL_WEIGHT_NAME,
                "arc": "save/bert-base-uncased/bidirectional-prompt-tuning/arc"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-base-uncased/bidirectional-prompt-tuning/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-base-uncased/bidirectional-prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/bidirectional-prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME
            },
            "p-tuning-v2":{
                "race": "save/bert-base-uncased/p-tuning-v2/race"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/p-tuning-v2/race-h"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/p-tuning-v2/race-m"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/p-tuning-v2/race-c"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-base-uncased/p-tuning-v2/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-base-uncased/p-tuning-v2/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/p-tuning-v2/commonsense_qa"+MODEL_WEIGHT_NAME
            },
            "lora":{
                "race": "save/bert-base-uncased/lora/race"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/lora/race-h"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/lora/race-m"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/lora/race-c"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-base-uncased/lora/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-base-uncased/lora/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/lora/commonsense_qa"+MODEL_WEIGHT_NAME

            },
            "o-lora":{
                "race": "save/bert-base-uncased/o-lora/race"+MODEL_WEIGHT_NAME,
                "race-h": "save/bert-base-uncased/o-lora/race-h"+MODEL_WEIGHT_NAME,
                "race-m": "save/bert-base-uncased/o-lora/race-m"+MODEL_WEIGHT_NAME,
                "race-c": "save/bert-base-uncased/o-lora/race-c"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-base-uncased/o-lora/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-base-uncased/o-lora/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-base-uncased/o-lora/commonsense_qa"+MODEL_WEIGHT_NAME
            }
            
        },
        "bert-base-NER": {
            
        },
        "bert-large-uncased":{
            "prompt-tuning":{
                "race": "save/bert-large-uncased/prompt-tuning/race"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-large-uncased/prompt-tuning/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-large-uncased/prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-large-uncased/prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
            },
            "bidirectional-prompt-tuning":{
                "race": "save/bert-large-uncased/bidirectional-prompt-tuning/race"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-large-uncased/bidirectional-prompt-tuning/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-large-uncased/bidirectional-prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-large-uncased/bidirectional-prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
            },
            "p-tuning":{
                "race": "save/bert-large-uncased/p-tuning/race"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-large-uncased/p-tuning/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-large-uncased/p-tuning/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-large-uncased/p-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
            },
            "p-tuning-v2":{
                "race": "save/bert-large-uncased/p-tuning-v2/race"+MODEL_WEIGHT_NAME,
                "dream": "save/bert-large-uncased/p-tuning-v2/dream"+MODEL_WEIGHT_NAME,
                "sciq": "save/bert-large-uncased/p-tuning-v2/sciq"+MODEL_WEIGHT_NAME,
                "commonsense_qa": "save/bert-large-uncased/p-tuning-v2/commonsense_qa"+MODEL_WEIGHT_NAME,
            },
        },
        "qwen": {
            'Qwen2.5-0.5B':{

            },
            'Qwen2.5-1.5B':{
                "prompt-tuning":{
                    "race": "save/qwen/Qwen2.5-1.5B/prompt-tuning/race"+MODEL_WEIGHT_NAME,
                    "dream": "save/qwen/Qwen2.5-1.5B/prompt-tuning/dream"+MODEL_WEIGHT_NAME,
                    "sciq": "save/qwen/Qwen2.5-1.5B/prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                    "commonsense_qa": "save/qwen/Qwen2.5-1.5B/prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
                },
                "bidirectional-prompt-tuning":{
                    "race": "save/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/race"+MODEL_WEIGHT_NAME,
                    "dream": "save/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/dream"+MODEL_WEIGHT_NAME,
                    "sciq": "save/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/sciq"+MODEL_WEIGHT_NAME,
                    "commonsense_qa": "save/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
                },
                "p-tuning":{
                    "race": "save/qwen/Qwen2.5-1.5B/p-tuning/race"+MODEL_WEIGHT_NAME,
                    "dream": "save/qwen/Qwen2.5-1.5B/p-tuning/dream"+MODEL_WEIGHT_NAME,
                    "sciq": "save/qwen/Qwen2.5-1.5B/p-tuning/sciq"+MODEL_WEIGHT_NAME,
                    "commonsense_qa": "save/qwen/Qwen2.5-1.5B/p-tuning/commonsense_qa"+MODEL_WEIGHT_NAME,
                },
                "p-tuning-v2":{
                    "race": "save/qwen/Qwen2.5-1.5B/p-tuning-v2/race"+MODEL_WEIGHT_NAME,
                    "dream": "save/qwen/Qwen2.5-1.5B/p-tuning-v2/dream"+MODEL_WEIGHT_NAME,
                    "sciq": "save/qwen/Qwen2.5-1.5B/p-tuning-v2/sciq"+MODEL_WEIGHT_NAME,
                   "commonsense_qa": "save/qwen/Qwen2.5-1.5B/p-tuning-v2/commonsense_qa"+MODEL_WEIGHT_NAME
                }
                

            },
            'Qwen2.5-3B':{

            }
        },
        "llama": {
            "Llama-3.2-1B":{


            }   
        },
    },
    
    
    
    "logging_dir": {
        "bert-base-uncased": {
            "prompt-tuning":{
                "race": "logs/bert-base-uncased/prompt-tuning/race",
                "race-m": "logs/bert-base-uncased/prompt-tuning/race-m",
                "race-h": "logs/bert-base-uncased/prompt-tuning/race-h",
                "race-c": "logs/bert-base-uncased/prompt-tuning/race-c",
                "record": "logs/bert-base-uncased/prompt-tuning/record", # ReCoRD
                "multirc": "logs/bert-base-uncased/prompt-tuning/multirc",  #MultiRC
                "arc": "logs/bert-base-uncased/prompt-tuning/arc",   #ARC
                "dream": "logs/bert-base-uncased/prompt-tuning/dream", # DREAM
                "sciq": "logs/bert-base-uncased/prompt-tuning/sciq",
                "commonsense_qa": "logs/bert-base-uncased/prompt-tuning/commonsense_qa",
            },
            "prefix-tuning":{
                "race": "logs/bert-base-uncased/prefix-tuning/race",
                "race-m": "logs/bert-base-uncased/prefix-tuning/race-m",
                "race-h": "logs/bert-base-uncased/prefix-tuning/race-h",
                "race-c": "logs/bert-base-uncased/prefix-tuning/race-c",
                "record": "logs/bert-base-uncased/prefix-tuning/record",
                "multirc": "logs/bert-base-uncased/prefix-tuning/multirc",
                "arc": "logs/bert-base-uncased/prefix-tuning/arc",
                "dream": "logs/bert-base-uncased/prefix-tuning/dream",
                "sciq": "logs/bert-base-uncased/prefix-tuning/sciq",
                "commonsense_qa": "logs/bert-base-uncased/prefix-tuning/commonsense_qa",

            },
            "bidirectional-prompt-tuning":{
                "race": "logs/bert-base-uncased/bidirectional-prompt-tuning/race",
                "race-m": "logs/bert-base-uncased/bidirectional-prompt-tuning/race-m",
                "race-h": "logs/bert-base-uncased/bidirectional-prompt-tuning/race-h",
                "race-c": "logs/bert-base-uncased/bidirectional-prompt-tuning/race-c",
                "record": "logs/bert-base-uncased/bidirectional-prompt-tuning/record",
                "multirc": "logs/bert-base-uncased/bidirectional-prompt-tuning/multirc",
                "arc": "logs/bert-base-uncased/bidirectional-prompt-tuning/arc",
                "dream": "logs/bert-base-uncased/bidirectional-prompt-tuning/dream",
                "sciq": "logs/bert-base-uncased/bidirectional-prompt-tuning/sciq",
                "commonsense_qa": "logs/bert-base-uncased/bidirectional-prompt-tuning/commonsense_qa",
            },
            "p-tuning-v2":{
                "race": "logs/bert-base-uncased/p-tuning-v2/race",
                "race-h": "logs/bert-base-uncased/p-tuning-v2/race-h",
                "race-m": "logs/bert-base-uncased/p-tuning-v2/race-m",
                "race-c": "logs/bert-base-uncased/p-tuning-v2/race-c",
                "dream": "logs/bert-base-uncased/p-tuning-v2/dream",
                "sciq": "logs/bert-base-uncased/p-tuning-v2/sciq",
                "commonsense_qa": "logs/bert-base-uncased/p-tuning-v2/commonsense_qa"
            },
            "lora":{
                "race": "logs/bert-base-uncased/lora/race",
                "race-h": "logs/bert-base-uncased/lora/race-h",
                "race-m": "logs/bert-base-uncased/lora/race-m",
                "race-c": "logs/bert-base-uncased/lora/race-c",
                "dream": "logs/bert-base-uncased/lora/dream",
                "sciq": "logs/bert-base-uncased/lora/sciq",
                "commonsense_qa": "logs/bert-base-uncased/lora/commonsense_qa"

            },
            "o-lora":{
                "race": "logs/bert-base-uncased/o-lora/race",
                "race-h": "logs/bert-base-uncased/o-lora/race-h",
                "race-m": "logs/bert-base-uncased/o-lora/race-m",
                "race-c": "logs/bert-base-uncased/o-lora/race-c",
                "dream": "logs/bert-base-uncased/o-lora/dream",
                "sciq": "logs/bert-base-uncased/o-lora/sciq",
                "commonsense_qa": "logs/bert-base-uncased/o-lora/commonsense_qa"
            }
            
        },
        "bert-base-NER": {
            
        },
        "bert-large-uncased":{
            "prompt-tuning":{
                "race": "logs/bert-large-uncased/prompt-tuning/race",
                "dream": "logs/bert-large-uncased/prompt-tuning/dream",
                "sciq": "logs/bert-large-uncased/prompt-tuning/sciq",
                "commonsense_qa": "logs/bert-large-uncased/prompt-tuning/commonsense_qa",
            },
            "bidirectional-prompt-tuning":{
                "race": "logs/bert-large-uncased/bidirectional-prompt-tuning/race",
                "dream": "logs/bert-large-uncased/bidirectional-prompt-tuning/dream",
                "sciq": "logs/bert-large-uncased/bidirectional-prompt-tuning/sciq",
                "commonsense_qa": "logs/bert-large-uncased/bidirectional-prompt-tuning/commonsense_qa",
            },
            "p-tuning":{
                "race": "logs/bert-large-uncased/p-tuning/race",
                "dream": "logs/bert-large-uncased/p-tuning/dream",
                "sciq": "logs/bert-large-uncased/p-tuning/sciq",
                "commonsense_qa": "logs/bert-large-uncased/p-tuning/commonsense_qa",
            },
            "p-tuning-v2":{
                "race": "logs/bert-large-uncased/p-tuning-v2/race",
                "dream": "logs/bert-large-uncased/p-tuning-v2/dream",
                "sciq": "logs/bert-large-uncased/p-tuning-v2/sciq",
                "commonsense_qa": "logs/bert-large-uncased/p-tuning-v2/commonsense_qa",
            },
        },
        "qwen": {
            'Qwen2.5-0.5B':{

            },
            'Qwen2.5-1.5B':{
                "prompt-tuning":{
                    "race": "logs/qwen/Qwen2.5-1.5B/prompt-tuning/race",
                    "dream": "logs/qwen/Qwen2.5-1.5B/prompt-tuning/dream",
                    "sciq": "logs/qwen/Qwen2.5-1.5B/prompt-tuning/sciq",
                    "commonsense_qa": "logs/qwen/Qwen2.5-1.5B/prompt-tuning/commonsense_qa",
                },
                "bidirectional-prompt-tuning":{
                    "race": "logs/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/race",
                    "dream": "logs/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/dream",
                    "sciq": "logs/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/sciq",
                    "commonsense_qa": "logs/qwen/Qwen2.5-1.5B/bidirectional-prompt-tuning/commonsense_qa",
                },
                "p-tuning":{
                    "race": "logs/qwen/Qwen2.5-1.5B/p-tuning/race",
                    "dream": "logs/qwen/Qwen2.5-1.5B/p-tuning/dream",
                    "sciq": "logs/qwen/Qwen2.5-1.5B/p-tuning/sciq",
                    "commonsense_qa": "logs/qwen/Qwen2.5-1.5B/p-tuning/commonsense_qa",
                },
                "p-tuning-v2":{
                    "race": "logs/qwen/Qwen2.5-1.5B/p-tuning-v2/race",
                    "dream": "logs/qwen/Qwen2.5-1.5B/p-tuning-v2/dream",
                    "sciq": "logs/qwen/Qwen2.5-1.5B/p-tuning-v2/sciq",
                   "commonsense_qa": "logs/qwen/Qwen2.5-1.5B/p-tuning-v2/commonsense_qa"
                }
                

            },
            'Qwen2.5-3B':{

            }
        },
        "llama": {
            "Llama-3.2-1B":{


            }   
        },
        
        
    },
    
    
   
    "models":{
        "bert-base-uncased": {
            "model_name": "bert-base-uncased",
            "max_length": 512,
            "num_labels": 4,
            "hidden_dim": 768,
            "model_path": f"{MODEL_BASE_PATH}bert-base-uncased"
        },
        "bert-base-NER":{
            "model_name": "bert-base-NER",
            "max_length": 512,
            "num_labels": 4,
            "hidden_dim": 768,
            "model_path": f"{MODEL_BASE_PATH}bert-base-NER"
        },
        
        "bert-large-uncased":{
            "model_name": "bert-large-uncased",
            "max_length": 512,
            "num_labels": 4,
            "hidden_dim": 1024,
            "model_path": f"{MODEL_BASE_PATH}bert-large-uncased"
        },
    
        "qwen":{
            "Qwen2.5-0.5B":{
                "model_name": "Qwen2.5-0.5B",
                "max_length": 32768,
                "num_labels": 4,
                "hidden_dim": 896,
                "model_type": "qwen2",
                "model_path": f"{MODEL_BASE_PATH}Qwen2.5-0.5B"
            },
            "Qwen2.5-1.5B":{
                "model_name": "Qwen2.5-1.5B",
                "max_length": 131072,
                "num_labels": 4,
                "hidden_dim": 1536,
                "model_type": "qwen2",
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-3B":{
                "model_name": "Qwen2.5-3B",
                "max_length": 32768,
                "num_labels": 4,
                "hidden_dim": 2048,
                "model_type": "qwen2",
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-7B":{
                "model_name": "Qwen2.5-7B",
                "max_length": 131072,
                "num_labels": 1,
                "hidden_dim": 3584,
                "model_type": "qwen2",
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-72B":{
                "model_name": "Qwen2.5-72B",
                "max_length": 131072,
                "num_labels": 1,
                "hidden_dim": 8192,
                "model_type": "qwen2",
                "model_path": "models/qwen_model.pth"
            }
           
        },
        
        "llama":{
            "MobiLlama-05B":{
                "model_name": "MobiLlama-05B",
                "max_length": 2048,
                "num_labels": 1,
                "hidden_dim": 2048,
                "model_type": "llama",
                "model_path": "models/llama_model.pth"
            },
            "Llama-3.2-1B":{
                "model_name": "Llama-3.2-1B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": f"{MODEL_BASE_PATH}Llama-3.2-1B"

            },
            "Llama-3.2-3B":{
                "model_name": "Llama-3.2-3B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": f"{MODEL_BASE_PATH}Llama-3.2-3B"
            },
            "Llama-3.1-8B":{
                "model_name": "Llama-3.1-8B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/llama_model.pth"
            },
            "Llama-3.1-70B":{
                "model_name": "Llama-3.1-70B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/llama_model.pth"
            }
        }
    },
    
    "datasets":{
        "race-c":"./data/race-c",  # tasksource/race-c
        "race":"./data/race",   # ehovy/race
        "race-m":"./data/race",
        "race-h":"./data/race",
        "record": {
            "all" : "./data/record",
            "train": "./data/record/train.json",
            "validation": "./data/record/dev.json"
            }, # ReCoRD  Reading Comprehension with Commonsense Reasoning Dataset
        "multirc": {
                    "all":"./data/mutlirc-v2",
                    "train":"./data/mutlirc-v2/train.json",
                    "validation":"./data/mutlirc-v2/dev.json"
            } , #MultiRC
        "arc": "./data/ai2_arc",   #ARC
        "dream": {
            "all": "./data/dream", # DREAM   https://github.com/nlpdata/dream
            "train" : "./data/dream/train.json",
            "validation": "./data/dream/dev.json",
            "test": "./data/dream/test.json"
        },
        "super_glue" : "./data/super_glue",
        "commonsense_qa": "./data/commonsense_qa",
        "sciq": "./data/sciq",
    },
    "classes":{
        "race-c":"",
        "race":"",
    }
}



MAX_LENGTH = Config['models']['bert-large-uncased']['max_length']
HIDDEN_DIM = Config['models']['bert-large-uncased']['hidden_dim']













def get_cpu_info():  
    """  
    获取CPU相关信息的函数  
    
    Returns:  
        dict: 包含CPU信息的字典，包括：  
            - logical_cores: 逻辑核心数（包含超线程）  
            - physical_cores: 物理核心数  
            - cpu_usage: 当前CPU使用率  
            - available_cores: 可用的CPU核心数  
    """  
    # 获取逻辑CPU核心数（包含超线程）  
    logical_cores = multiprocessing.cpu_count()  
    
    # 获取物理CPU核心数（不包含超线程）  
    physical_cores = psutil.cpu_count(logical=False)  
    
    # 获取CPU使用率（百分比）  
    cpu_usage = psutil.cpu_percent(interval=1)  
    
    # 获取当前进程可用的CPU核心数  
    # 这会考虑环境变量中的限制  
    if 'SLURM_CPUS_PER_TASK' in os.environ:  # 如果在SLURM集群环境中  
        available_cores = int(os.environ['SLURM_CPUS_PER_TASK'])  
    elif 'OMP_NUM_THREADS' in os.environ:  # 如果设置了OpenMP线程数  
        available_cores = int(os.environ['OMP_NUM_THREADS'])  
    else:  
        available_cores = logical_cores  

    return {  
        'logical_cores': logical_cores,  
        'physical_cores': physical_cores,  
        'cpu_usage': cpu_usage,  
        'available_cores': available_cores  
    }  

def recommend_num_workers(cpu_info):  
    """  
    根据CPU信息推荐合适的工作进程数  
    
    Args:  
        cpu_info (dict): CPU信息字典  
    
    Returns:  
        int: 推荐的工作进程数  
    """  
    # 获取可用的CPU核心数  
    available_cores = cpu_info['available_cores']  
    
    # 保留一个核心给系统运行其他任务  
    # 如果核心数大于4，则预留一个核心  
    # 如果核心数小于等于4，则使用总核心数的3/4  
    if available_cores > 4:  
        recommended_workers = available_cores - 1  
    else:  
        recommended_workers = max(1, int(available_cores * 0.75))  
    
    return recommended_workers  



NUM_CPU_PROCESSES = 0
try:  
    # 获取CPU信息  
    cpu_info = get_cpu_info()  
    print("\n=== CPU 信息 ===") 
    print(f"当前环境可用核心数: {cpu_info['available_cores']}")  
    # 获取推荐的工作进程数  
    NUM_CPU_PROCESSES = recommend_num_workers(cpu_info)  
    print(f"\n推荐的工作进程数: {NUM_CPU_PROCESSES}")  

    print("\n=== 使用建议 ===")  
    print("1. 对于CPU密集型任务（如数据预处理）:")  
    print(f"   - 建议使用 num_proc={NUM_CPU_PROCESSES}") 
except Exception as e:  
        print(f"获取CPU信息时发生错误: {str(e)}")  
        
        
print("NUM_CPU_PROCESSES = ", NUM_CPU_PROCESSES)




class BaasPromptConfig(ABC):
    """Abstract class for a BaasPrompt configuration that can be saved to and loaded from a json file."""
    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)


class TrainConfig(BaasPromptConfig):
    """Configuration for training a model."""

    def __init__(self,
                 device: str = None,
                 per_gpu_train_batch_size: int = 8,
                 n_gpu: int = 1,
                 num_train_epochs: int = 3,
                 max_steps: int = -1,
                 gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0,
                 learning_rate: float = 5e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1,
                 alpha: float = 0.9999):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param alpha: the alpha parameter for auxiliary language modeling
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha




class EvalConfig(BaasPromptConfig):
    """Configuration for evaluating a model."""

    def __init__(self,
                 device: str = None,
                 n_gpu: int = 1,
                 per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = ['accuracy']):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics





class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 max_length: int,
                 label_list: List[str],
                 pattern_id: int = 0,  # indicate the length of the continuous prompt tokens
                 cache_dir: str = None,
                 output_dir=None,
                 embedding_size=128,
                 prompt_encoder_type="lstm", # ["lstm", "sparse-attention"]
                 eval_every_step=20):

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_length = max_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.embedding_size = embedding_size
        self.prompt_encoder_type = prompt_encoder_type
        self.eval_every_step = eval_every_step
        
        


class BudgetSchedulerConfig(object):
    def __init__(self,
                min_rank=10,  
                max_rank=100,  
                alpha=0.01,  
                beta=0.9,  
                H_max=1.0,  
                A=0.1,  
                omega=0.1,  
                phi=0,  # 相位
                lambda_=0.001,  
                t_scale=1000,
                T0 = 1.0,  # 初始温度  
                tau = 0.9,  # softmax temprature
                eta = 0.1,  # 温度调节率  
                min_temperature = 0.1,  
                max_temperature = 5.0,  
                gamma = 0.5,  # 探索项对温度的影响系数
                k = 5,
                theta = 0.8,
                epsilon = 1e-8,
                
                 ):
        '''
        Args
        :param min_rank: 最小的阶数（reasoning step数）一般设为min(n_step)
        :param max_rank: 最大的阶数（reasoning step数）一般设为max(n_step)
        :param alpha: 学习率
        :param beta: 动量系数
        :param H_max: 熵的归一化因子
        :param A: 探索振幅
        :param omega: 探索频率
        :param phi: 相位偏移
        :param lambda_: 探索衰减率
        :param t_scale: 时间步t的scaling系数
        
        :param T0: 初始温度, 用于计算每个时间步的温度 T^{(t)} = T_0*(1 + gamma*E(t))
        :param tau: softmax temprature, 它控制概率分布的"软硬程度" 0.1 ~ 1.0
                τ = 1.0：标准Softmax，适用于一般情况
                τ < 1.0：使分布更"尖锐"(sharper)，突出高满意度的差异, 适用于n_step较小
                τ > 1.0：使分布更"平滑"(smoother)，减小满意度差异，适用于n_step较大
        :param: eta:  温度调节率  
        :param: min_temperature 
        :param: max_temperature 
        :param: gamma:  探索项对温度的影响系数
        
        :param: k  Top-k for sparse attention  
        :param: theta  # 满意度阈值 
        :param: epsilon  # 噪声系数
        
        '''
        
        self.min_rank=min_rank
        self.max_rank=max_rank
        self.alpha=alpha
        self.beta=beta
        self.H_max=H_max
        self.A=A
        self.omega=omega
        self.phi=phi
        self.lambda_=lambda_
        self.t_scale=t_scale
        
        # 温度相关参数  
        self.T0 = T0
        self.tau = tau
        self.eta = eta
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.gamma = gamma 
        
        self.k = k  # Top-k for sparse attention 
        self.epsilon = epsilon
        self.theta = theta