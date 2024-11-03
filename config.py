import torch
import platform

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


Config = {
    "output_dir":"./output",
    "logging_dir":"./logs",
    "vocab":"data/chars.txt",
    "train_data_path": "data/train.csv",
    "test_data_path": "data/test.csv",
    "submission_path": "submission.csv",
    

    
    "seed": 42,
    "hidden_size": 256,
    "num_layers": 4,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "num_epochs": 2,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model.pth",
    
    
    "save_model_dir":{
        "bert-base-uncased": {
            "prompt-tuning":{
                "race": "save/bert-base-uncased/prompt-tuning/race",
                "race-m": "save/bert-base-uncased/prompt-tuning/race-m",
                "race-h": "save/bert-base-uncased/prompt-tuning/race-h",
                "race-c": "save/bert-base-uncased/prompt-tuning/race-c",
                "record": "save/bert-base-uncased/prompt-tuning/record", # ReCoRD
                "multirc": "save/bert-base-uncased/prompt-tuning/multirc",  #MultiRC
                "arc": "save/bert-base-uncased/prompt-tuning/arc",   #ARC
                "dream": "save/bert-base-uncased/prompt-tuning/dream", # DREAM
            },
            "prefix-tuning":{
                "race": "save/bert-base-uncased/prefix-tuning/race",
                "race-m": "save/bert-base-uncased/prefix-tuning/race-m",
                "race-h": "save/bert-base-uncased/prefix-tuning/race-h",
                "race-c": "save/bert-base-uncased/prefix-tuning/race-c",
                "record": "save/bert-base-uncased/prefix-tuning/record",
                "multirc": "save/bert-base-uncased/prefix-tuning/multirc",
                "arc": "save/bert-base-uncased/prefix-tuning/arc",
                "dream": "save/bert-base-uncased/prefix-tuning/dream",

            },
            "bidirectional-prompt-tuning":{
                "race": "save/bert-base-uncased/bidirectional-prompt-tuning/race",
                "race-m": "save/bert-base-uncased/bidirectional-prompt-tuning/race-m",
                "race-h": "save/bert-base-uncased/bidirectional-prompt-tuning/race-h",
                "race-c": "save/bert-base-uncased/bidirectional-prompt-tuning/race-c",
                "record": "save/bert-base-uncased/bidirectional-prompt-tuning/record",
                "multirc": "save/bert-base-uncased/bidirectional-prompt-tuning/multirc",
                "arc": "save/bert-base-uncased/bidirectional-prompt-tuning/arc",
                "dream": "save/bert-base-uncased/bidirectional-prompt-tuning/dream",

            },
            "p-tuning-v2":{
                "race": "save/bert-base-uncased/p-tuning-v2/race",
                "race-h": "save/bert-base-uncased/p-tuning-v2/race-h",
                "race-m": "save/bert-base-uncased/p-tuning-v2/race-m",
                "race-c": "save/bert-base-uncased/p-tuning-v2/race-c",

            },
            "lora":{
                "race": "save/bert-base-uncased/lora/race",
                "race-h": "save/bert-base-uncased/lora/race-h",
                "race-m": "save/bert-base-uncased/lora/race-m",
                "race-c": "save/bert-base-uncased/lora/race-c",

            },
            "o-lora":{
                "race": "save/bert-base-uncased/o-lora/race",
                "race-h": "save/bert-base-uncased/o-lora/race-h",
                "race-m": "save/bert-base-uncased/o-lora/race-m",
                "race-c": "save/bert-base-uncased/o-lora/race-c",

            }
            
        },
        "bert-base-NER": {
            
        },
        "qwen": {
            'Qwen2.5-0.5B':{

            },
            'Qwen2.5-1.5B':{

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
            "model_path": f"{MODEL_BASE_PATH}bert-base-uncased"
        },
        "bert-base-NER":{
            "model_name": "bert-base-NER",
            "max_length": 512,
            "num_labels": 4,
            "model_path": f"{MODEL_BASE_PATH}bert-base-NER"
        },
        
        "bert-large-uncased":{
            "model_name": "bert-large-uncased",
            "max_length": 1024,
            "num_labels": 4,
            "model_path": f"{MODEL_BASE_PATH}bert-large-uncased"
        },
    
        "qwen":{
            "Qwen2.5-0.5B":{
                "model_name": "Qwen2.5-0.5B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": f"{MODEL_BASE_PATH}Qwen2.5-0.5B"
            },
            "Qwen2.5-1.5B":{
                "model_name": "Qwen2.5-1.5B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-3B":{
                "model_name": "Qwen2.5-3B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-7B":{
                "model_name": "Qwen2.5-7B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/qwen_model.pth"
            },
            "Qwen2.5-72B":{
                "model_name": "Qwen2.5-72B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/qwen_model.pth"
            }
           
        },
        
        "llama":{
            "MobiLlama-05B":{
                "model_name": "MobiLlama-05B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/llama_model.pth"
            },
            "Llama-3.2-1B":{
                "model_name": "Llama-3.2-1B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/llama_model.pth"

            },
            "Llama-3.2-3B":{
                "model_name": "Llama-3.2-3B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": "models/llama_model.pth"  
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
        "record": "./data/record", # ReCoRD
        "multirc": {
                    "all":"./data/mutlirc-v2",
                    "train":"./data/mutlirc-v2/train_456-fixedIds.json",
                    "validation":"./data/mutlirc-v2/dev_83-fixedIds.json"} , #MultiRC
        "arc": "./data/ai2_arc",   #ARC
        "dream": "./data/dream", # DREAM   https://github.com/nlpdata/dream
    },
    "classes":{
        "race-c":"",
        "race":"",
    }
}