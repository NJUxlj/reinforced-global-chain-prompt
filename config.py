import torch
import platform
import multiprocessing  
import psutil  
import os 

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
