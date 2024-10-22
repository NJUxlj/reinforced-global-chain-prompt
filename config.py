import torch

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
    "batch_size": 32,
    "num_epochs": 100,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model.pth",
    
    
    "save_model_dir":{
        "bert-base-uncased": {
            "prompt-tuning":{
                "race": "save/bert-base-uncased/prompt-tuning/race",
                "race-c": "save/bert-base-uncased/prompt-tuning/race-c",
                "mnli": "save/bert-base-uncased/prompt-tuning/mnli",
                "mrpc": "save/bert-base-uncased/prompt-tuning/mrpc",
            },
            "bidirectional-prompt-tuning":{
                "race": "save/bert-base-uncased/bidirectional-prompt-tuning/race",
                "race-c": "save/bert-base-uncased/bidirectional-prompt-tuning/race-c",
                "mnli": "save/bert-base-uncased/bidirectional-prompt-tuning/mnli",
                "mrpc": "save/bert-base-uncased/bidirectional-prompt-tuning/mrpc",
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
            "num_labels": 1,
            "model_path": r"D:\pre-trained-models\bert-base-uncased"
        },
        "bert-base-NER":{
            "model_name": "bert-base-NER",
            "max_length": 512,
            "num_labels": 1,
            "model_path": r"D:\pre-trained-models\bert-base-NER"
        },
        
        "qwen":{
            "Qwen2.5-0.5B":{
                "model_name": "Qwen2.5-0.5B",
                "max_length": 2048,
                "num_labels": 1,
                "model_path": r"D:\pre-trained-models\Qwen2.5-0.5B"
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
    },
    "classes":{
        "race-c":"",
        "race":"",
    }
}