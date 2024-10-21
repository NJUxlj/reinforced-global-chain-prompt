import torch

Config = {
    "vocab":"data/chars.txt",
    "hidden_size": 256,
    "num_layers": 4,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 100,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model.pth",
    "train_data_path": "data/train.csv",
    "test_data_path": "data/test.csv",
    "submission_path": "submission.csv",
    "seed": 42,
    "models":{
        "bert": {
            "model_name": "bert-base-uncased",
            "max_length": 512,
            "num_labels": 1,
            "model_path": "models/bert_model.pth"
        },
        
        "qwen":{
            "model_name": "Qwen/Qwen-14B-Chat",
            "max_length": 2048,
            "num_labels": 1,
            "model_path": "models/qwen_model.pth"
        },
        
        "llama":{
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "max_length": 2048,
            "num_labels": 1,
            "model_path": "models/llama_model.pth"
        }
    }
}