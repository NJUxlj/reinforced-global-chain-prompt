import os
import torch
import logging
import evaluate
import random
import argparse
import numpy as np
from itertools import product
from config import Config

import datetime


from gensim.models import Word2Vec, KeyedVectors  
from typing import List, Union, Optional  

import torch.distributed as dist  
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP 

from torch.utils.data import DataLoader, DistributedSampler

from datasets import (
    Dataset,
    load_dataset
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    BertTokenizerFast,
    T5TokenizerFast,
    Qwen2TokenizerFast,
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    
    BertForSequenceClassification,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    RobertaForSequenceClassification,
    GPT2ForSequenceClassification,
)

from wrapper import *

from config import Config

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import os  
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
os.environ['TORCH_USE_CUDA_DSA'] = '1' 


def setup_cuda_debug_environment():  
    """设置调试环境"""  
    import torch  
    
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32以获得更精确的错误信息  
    torch.backends.cudnn.deterministic = True      # 使用确定性算法  
    torch.backends.cudnn.benchmark = False         # 禁用基准测试优化  
    
    import os  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
    os.environ['TORCH_USE_CUDA_DSA'] = '1' 
    
    print("=== Debug Environment Setup ===")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    print(f"CUDA version: {torch.version.cuda}")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"TORCH_USE_CUDA_DSA: {os.getenv('TORCH_USE_CUDA_DSA')}")  
    print(f"Current device: {torch.cuda.current_device()}")  
    print(f"Device name: {torch.cuda.get_device_name()}")  
    print("===========================")  
# 在关键操作点添加同步检查  
def debug_cuda_sync(name="operation"):  
    try:  
        torch.cuda.synchronize()  
        print(f"{name} completed successfully")  
    except RuntimeError as e:  
        print(f"CUDA error detected at {name}: {e}")  
        raise  
    
def create_position_ids_safe(attention_mask, model_max_length=512, prefix_length:int=0,padding_idx=1):  
    """  
    安全地创建position_ids，确保所有索引都在有效范围内  
    
    Args:  
        attention_mask: [batch_size, seq_length]  
        model_max_length: 模型支持的最大序列长度  
        padding_idx: padding token的position id值  
    """  
    batch_size, seq_length = attention_mask.shape  
    device = attention_mask.device  
    
    effective_length = model_max_length  # 414
    
    # 创建基础position_ids (从padding_idx + 1开始)  
    position_ids = torch.arange(  
        prefix_length,  
        effective_length+prefix_length,  
        dtype=torch.long,  
        device=device  
    )  
    
    # 扩展到batch维度  
    position_ids = position_ids.unsqueeze(0).expand(1, -1)  
    
    # 处理padding位置  
    attention_mask_truncated = attention_mask[:, :effective_length] 
    
    # 根据 attention_mask_truncated 来处理 position_ids，
    # 如果对应位置是 1（非 padding），则使用原来的 position_ids，如果是 0（padding），则使用 padding_idx。
    # position_ids = position_ids * attention_mask_truncated + padding_idx * (1 - attention_mask_truncated)  
    
    # 如果序列长度小于预期长度，进行padding 
    # 如果原序列长度大于有效长度，需要补充padding 
    # if seq_length > effective_length:  
    #     padding = torch.full(  
    #         (batch_size, seq_length - effective_length),  
    #         padding_idx,  
    #         dtype=torch.long,  
    #         device=device  
    #     )  
    #     position_ids = torch.cat([position_ids, padding], dim=1)  
    
    return position_ids  


def forward_with_safe_position_ids(model, batch, debug=False):  
    """  
    使用安全的position_ids进行前向传播  
    """  
    try:  
        # 获取模型的配置  
        max_position_embeddings = get_max_length_from_model(model)
        
        # 确保所有输入在同一设备上  
        device = next(model.parameters()).device  
        batch = {k: v.to(device) for k, v in batch.items()}  
        
        # 获取attention_mask  
        attention_mask = batch.get('attention_mask')  
        if attention_mask is None:  
            raise ValueError("Batch must contain attention_mask")  
        
        # 创建安全的position_ids  
        position_ids = create_position_ids_safe(  
            attention_mask,  
            model_max_length=max_position_embeddings  
        )  
        
        if debug:  
            print("\n=== Debug Information ===")  
            print(f"Attention mask shape: {attention_mask.shape}")  
            print(f"Position IDs shape: {position_ids.shape}")  
            print(f"Position IDs range: [{position_ids.min()}, {position_ids.max()}]")  
            print(f"Max position embeddings: {max_position_embeddings}")  
        
        # 验证position_ids  
        assert position_ids.max() < max_position_embeddings, \
            f"Position IDs max value ({position_ids.max()}) exceeds model limit ({max_position_embeddings})"  
        assert position_ids.min() >= 0, \
            f"Position IDs contain negative values: {position_ids.min()}"  
        
        # 前向传播  
        outputs = model(**batch, position_ids=position_ids)  
        return outputs  
        
    except Exception as e:  
        print("\n=== Error Information ===")  
        print(f"Error type: {type(e).__name__}")  
        print(f"Error message: {str(e)}")  
        
        if 'attention_mask' in locals():  
            print(f"\nAttention mask info:")  
            print(f"Shape: {attention_mask.shape}")  
            print(f"Device: {attention_mask.device}")  
            print(f"Values range: [{attention_mask.min()}, {attention_mask.max()}]")  
        
        if 'position_ids' in locals():  
            print(f"\nPosition IDs info:")  
            print(f"Shape: {position_ids.shape}")  
            print(f"Device: {position_ids.device}")  
            print(f"Values range: [{position_ids.min()}, {position_ids.max()}]")  
        
        raise  
def print_training_info(config):
    print("\n\n****************** 打印训练信息 ********************************")
    print("********************** ----------- ********************************")
    print(f"PEFT tuning method: {config.peft_method}")
    print(f"model_name: {config.model_name}")
    print(f"dataset_name: {config.dataset_name}")
    print(f"model's max sequence length = {config.max_seq_length}")
    
    if hasattr(config,'prefix_length'):
        print(f"prefix_length: {config.prefix_length}")
        
        
    print("***********************************************************\n\n")


def check_model_config(model, config):  
    """检查模型配置是否合理"""  
    print("\n=== Model Configuration Check ===")  
    print(f"Model type: {model.config.model_type}")  
    print(f"Max position embeddings: {model.config.max_position_embeddings}")  
    print(f"Hidden size: {model.config.hidden_size}")  
    print(f"Vocab size: {model.config.vocab_size}")  
    print(f"Prefix length: {config.prefix_length}")  
    print(f"Max sequence length: {config.max_seq_length}")  
    print(f"Batch size: {config.batch_size}")  
    
    # 检查关键参数  
    assert config.prefix_length < model.config.max_position_embeddings, \
        f"Prefix length ({config.prefix_length}) too large for model max position embeddings ({model.config.max_position_embeddings})"  
    
    assert config.max_seq_length <= model.config.max_position_embeddings, \
        f"Max sequence length ({config.max_seq_length}) exceeds model max position embeddings ({model.config.max_position_embeddings})"  
    
    assert config.prefix_hidden_size == model.config.hidden_size, \
        f"Prefix hidden size ({config.prefix_hidden_size}) must match model hidden size ({model.config.hidden_size})"  


def get_dataset_path_by_name(dataset_name='race'):
    
    if dataset_name == "race":
        dataset_path = Config["datasets"][dataset_name]
    elif dataset_name == 'dream':
        dataset_path = Config["datasets"][dataset_name]['all']
    elif dataset_name == 'sciq':
        dataset_path = Config["datasets"][dataset_name]
    elif dataset_name == 'commonsense_qa':
        dataset_path = Config["datasets"][dataset_name]
    else:
        raise ValueError("dataset name not supported")
    
    return dataset_path

def get_model_name_using_model(model):
    '''
    
    use the model object's config file to retrieve the model name, e.g. bert-base-uncased
    '''
    
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
        
    config = model.config  
    # 尝试直接获取模型的名称  
    if hasattr(config, 'name_or_path') and config.name_or_path is not None:  
        # 使用 os.path.basename 提取路径中的模型名称  
        model_name = os.path.basename(config.name_or_path)  
        return model_name  
    # 根据模型类型和隐藏层大小推断模型名称  
    if config.model_type == "bert":  
        if config.hidden_size == 768:  
            return "bert-base-uncased"  
        elif config.hidden_size == 1024:  
            return "bert-large-uncased"  
    elif config.model_type == "roberta":  
        if config.hidden_size == 768:  
            return "roberta-base"  
        elif config.hidden_size == 1024:  
            return "roberta-large"  
    elif config.model_type == "llama":  
        if config.hidden_size == 4096:  
            return "meta-llama/Llama-2-13b-hf"  
        elif config.hidden_size == 5120:  
            return "meta-llama/Llama-2-70b-hf"  
    elif config.model_type == "qwen2":  
        if config.hidden_size == 896:  
            return "Qwen2.5-0.5B"  
        elif config.hidden_size == 1536:  
            return "Qwen2.5-1.5B"  
        elif config.hidden_size == 2048:
            return "Qwen2.5-3B"
        elif config.hidden_size == 3584:
            return "Qwen2.5-7B"
    elif config.model_type == "gpt2":
        if config.n_embd == 768:
            return "gpt2"
        elif config.n_embd == 1024:
            return "gpt2-medium"
        elif config.n_embd == 1280:
            return "gpt2-large"
        elif config.n_embd== 1600:
            return "gpt2-xl"
    else:  
        # 无法匹配已知模型，返回未知模型提示  
        raise ValueError("unknown model, please check your config, it should be [bert | llama | qwen2]") 

def get_base_model_using_model(model):
    """
    获取模型包装器的底层的基座模型对象

    """
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")

    try:
        if isinstance(model, AutoModel):
            model = model
        elif isinstance(model, PeftModel):  
            print("Info: Model is a PeftModel, getting the base model")  
            model = model.get_base_model() 
        elif isinstance(model, AutoModelForSequenceClassification):
            model = model.base_model
        elif isinstance(model, BertForSequenceClassification):
            model = model.bert
        elif isinstance(model, RobertaForSequenceClassification):
            model = model.roberta
        elif isinstance(model, Qwen2ForSequenceClassification):
            model = model.model
        elif isinstance(model, GPT2ForSequenceClassification):
            model = model.transformer
         
        else:
            raise ValueError(f"the passed model object is not either SequenceClassification model or AutoModel \
                The current model type = {model_type}")

    except:
        raise ValueError(f"Extracting base model failed, your current model type is {model_type}")

    return model

def get_hidden_size_using_config():
    pass

def get_hidden_size_by_model_name(model_name:str):
    pass

def get_hidden_size_using_model(model):
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)
    
    model_name = get_model_name_using_model(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")
    
    if hasattr(config,'hidden_size'):
        hidden_size = config.hidden_size
    elif hasattr(config, 'd_model'): # t5
        hidden_size = config.d_model
    elif hasattr(config, 'n_embd'): # gpt2
        hidden_size = config.n_embd
    else:
        raise ValueError(f"the passed model object does not have the attribute `hidden_size` \
            The current model type = {model_type}")
    print(f"model:{model_name}'s hidden_size = {hidden_size}")
    return hidden_size

def get_classifier_from_model(model)-> nn.Module:  
    """  
    获取预训练模型的分类器  
    
    Args:  
        model : AutoModelForSequenceClassification or BertForSequenceClassification
        num_labels (int): 分类标签数量  
    
    Returns:  
        nn.Module: 分类器模块  
    """  
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    # 获取分类器  
    if hasattr(model, 'classifier'):  
        # BERT、RoBERTa 等模型的分类器  
        classifier = model.classifier  
        print(f"分类器类型: {type(classifier).__name__}")
        
    elif hasattr(model, 'score'):   # qwen2, gpt2
        # 某些模型可能使用 score 作为分类器名称  
        classifier = model.score  
    else:  
        raise AttributeError("无法找到模型的分类器层")  
    
    # 打印分类器信息  
    print("分类器结构：")  
    print(classifier)  
    
    in_features=None
    out_features=None
    if hasattr(classifier, 'dense'):
        in_features = classifier.dense.in_features
        print("这是一个RobertaClassificationHead，需要通过dense层获取输入维度")
    else:
        in_features = classifier.in_features
        
    if hasattr(classifier, 'out_proj'):
        out_features = classifier.out_proj.out_features
        print("这是一个RobertaClassificationHead，需要通过out_proj层获取输出维度")
    else:
        out_features = classifier.out_features
        
        
    print(f"\n分类器输入维度: {in_features}")  
    print(f"分类器输出维度: {out_features}") 
    
    # 示例：直接使用分类器进行前向传播  
    # batch_size = 4  
    # hidden_size = classifier.in_features  
    
    # 模拟来自BERT的特征输出  
    # dummy_features = torch.randn(batch_size, hidden_size)  
    
    # # 直接使用分类器进行预测  
    # with torch.no_grad():  
    #     outputs = classifier(dummy_features)  
        
    # print(f"\n分类器输出形状: {outputs.shape}")  
    # print("分类器输出示例：")  
    # print(outputs)   
    
    
    print("\n分类器的可训练参数：")  
    for name, param in classifier.named_parameters():  
        print(f"{name}: {param.shape}")  
        
    return classifier 

def get_max_length_from_model(model):  
    """  
    获取模型的最大序列长度  
    model: 既可以base model， 也可以是特定任务model
    
    """  
    # 处理被Accelerator(DDP)包装的模型  
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    if hasattr(model, "config"):
        config = model.config  
    else:
        raise ValueError('your model object is not properly defined ... since we can not find a `config` attribute')
    
    # 首先尝试从config中直接获取max_position_embeddings  
    if hasattr(config, 'max_position_embeddings'):  
        return config.max_position_embeddings  
    
    # 如果没有max_position_embeddings，尝试获取max_sequence_length  
    elif hasattr(config, 'max_sequence_length'):  
        return config.max_sequence_length  
    
    elif hasattr(config, 'n_positions'):  
        return config.n_positions
    
    elif hasattr(config, 'n_ctx'):  
        return config.n_ctx
    
    else:
        raise ValueError("Error model object, please check your config, it should have either [max_position_embeddings | max_sequence_length]") 



def print_model_info(model:AutoModelForSequenceClassification):  
    """打印模型的详细信息"""  
    
    
    print("\n=== Model Classification Head Information ===")  
    
    # 1. 打印分类器的结构  
    print("\nClassifier Architecture:")  
    print(model.classifier)  
    
    # 2. 打印分类器中dense层的权重形状  
    dense_weight = model.classifier.dense.weight  
    print("\nDense Layer Weight Shape:", dense_weight.shape)  
    
    # 3. 打印分类器中out_proj层的权重形状  
    out_proj_weight = model.classifier.out_proj.weight  
    print("Output Projection Weight Shape:", out_proj_weight.shape)  
    
    # 4. 打印整个模型的参数数量  
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"\nTotal Parameters: {total_params:,}")  
    print(f"Trainable Parameters: {trainable_params:,}")  
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%") 



def get_vocab_embeddings_from_model(model, token_ids:torch.LongTensor):
    '''
     model is a pretrained model
     
     token_ids: a tensor of shape (num_tokens, 1)
     
     return a tensor of shape (num_tokens, 1, hidden_size)
    '''
    
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    if hasattr(model, 'embeddings'):  
        return model.embeddings(token_ids)  
    elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):  
        return model.bert.embeddings(token_ids)  
    elif hasattr(model, 'roberta') and hasattr(model.roberta, 'embeddings'):  
        return model.roberta.embeddings(token_ids)  
    elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'embeddings'):  
        return model.distilbert.embeddings(token_ids)   
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'embeddings'):  
        return model.base_model.embeddings(token_ids)  
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  # qwen2
        return model.model.embed_tokens(token_ids)  
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte(token_ids)
    else:  
        raise AttributeError(f"Can not find the embedding layer in the model. Please check the model type {type(model).__name__}.") 


def get_word_embeddings_from_model(model):
    
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    try:
        if hasattr(model, 'embeddings'):  
            return model.embeddings.word_embeddings
        elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):  
            return model.bert.embeddings.word_embeddings
        elif hasattr(model, 'roberta') and hasattr(model.roberta, 'embeddings'):  
            return model.roberta.embeddings.word_embeddings 
        elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'embeddings'):  
            return model.distilbert.embeddings.word_embeddings 
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'embeddings'):  
            return model.base_model.embeddings.word_embeddings
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'): # qwen2
            return model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            return model.transformer.wte
        else:  
            raise AttributeError(f"Can not find the embedding layer in the model. Please check the model type {type(model).__name__}.") 

    except Exception as e:
        raise AttributeError(f"Can not find the word_embeddings in the model. Please check the model type {type(model).__name__}.")


def prepare_model_tokenizer(model_path, auto_model_class = AutoModel, tokenizer_path = None, num_labels = 2):
    '''
     return model, tokenizer
    '''
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, num_labels=num_labels)
    
    model = auto_model_class.from_pretrained(
        model_path, 
        config=config,
        )
    
    model_type = model.config.model_type
    
    tokenizer_class = AutoTokenizer
    if model_type == 'roberta':
        tokenizer_class = RobertaTokenizerFast
    elif model_type == 'bert':
        tokenizer_class = BertTokenizerFast
    elif model_type == 'qwen2':
        tokenizer_class = Qwen2TokenizerFast
    elif model_type == 't5':
        tokenizer_class = T5TokenizerFast
    elif model_type == 'gpt2':
        tokenizer_class = GPT2TokenizerFast
    else:
        print(f"The Fast Tokenizer of this model type {model_type} is not supported, We use the default AutoTokenzier")
    
    
    # if tokenizer_path is not None:
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True)
        
    
    
    model_config = model.config
    model_name_or_path = model_config.name_or_path
    
    print("\n********************** Model Information *********************")
    print("model_name_or_path = ", model_name_or_path)
    print(f"Model's current num_labels: {model.config.num_labels}") 
    print("model_type = ", model_type)
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    print("padding_side = ", padding_side)

    tokenizer = tokenizer_class.from_pretrained(model_path, padding_side=padding_side)    # , use_fast=True)
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")  
    
    print("pad_token_id = ", tokenizer.pad_token_id)
    # print("eos_token_id = ", tokenizer.eos_token_id)
    # print("model's hidden_size = ", model.config.hidden_size)
    # print("model's max_position_embeddings = ", model.config.max_position_embeddings)
    print("***********************************************\n\n")
    
    model.config.pad_token_id = tokenizer.pad_token_id  
    
    return model, tokenizer








def get_logger(name="my_logger", logging_dir = None,log_level="INFO"):
    # 配置基本日志设置：设置日志的最低显示级别和格式  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  

    
    # 确保日志目录存在  
    os.makedirs(logging_dir, exist_ok=True)  

    
    # 创建具体的日志文件路径  
    # 添加时间戳避免文件重名  
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    log_file_name = f"{name}_{timestamp}.log"  
    log_file_path = os.path.join(logging_dir, log_file_name)  
    
    # 创建logger对象  
    logger = logging.getLogger(name)  


    # 创建一个FileHandler处理对象，将日志输出到文件  
    file_handler = logging.FileHandler(log_file_path)  
    # 设置文件处理器的日志级别  
    file_handler.setLevel(logging.INFO)  
    
    # 创建控制台处理器  
    console_handler = logging.StreamHandler()  
    console_handler.setLevel(logging.INFO)

    # 创建格式化器，将格式对象与处理器关联  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    file_handler.setFormatter(formatter)  
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler) 
    logger.addHandler(file_handler)  
    
    return logger






def make_save_dirs(model_name:Union[List,str] = None, pt_method:Union[List, str] = None, dataset_name:Union[List, str] = None):
    '''
     生成所有的模型权重存储路径
     
     也可以指定某条具体路径的3段参数来生成单个目录
    
    '''
    
    model_name = ["bert-base-uncased", "bert-large-uncased", 'Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Qwen2.5-3B', 'Qwen2.5-3B'] 
    pt_method = ['p-tuning', 'prefix-tuning','prompt-tuning','bidirectional-prompt-tuning','p-tuning-v2', 'lora','o-lora','adalora']
    dataset_name = ['race', 'sciq', 'dream', 'commonsense_qa']
    # 检查是否有参数为 None , 只要model_name, pt_method, dataset_name 有一个参数为 None，就抛出异常
    if any(param is None for param in [model_name, pt_method, dataset_name]):  
        missing_params = []  
        if model_name is None:  
            missing_params.append("model_name")  
        if pt_method is None:  
            missing_params.append("pt_method")  
        if dataset_name is None:  
            missing_params.append("dataset_name")  
        raise ValueError(f"Missing model weights save path components: {', '.join(missing_params)}")  
    
    # 将字符串输入转换为列表  
    model_names = [model_name] if isinstance(model_name, str) else model_name  
    pt_methods = [pt_method] if isinstance(pt_method, str) else pt_method  
    dataset_names = [dataset_name] if isinstance(dataset_name, str) else dataset_name  
    
    # 检查是否有空字符串  
    def check_empty_strings(values: List[str], param_name: str):  
        if any(not val.strip() for val in values):  
            raise ValueError(f"Empty string found in {param_name}. All model weights save path components must be non-empty.")
    
    check_empty_strings(model_names, "model_name")  
    check_empty_strings(pt_methods, "pt_method")  
    check_empty_strings(dataset_names, "dataset_name")  
    
    
    base_path = "./save" 
    
    weights_file_name = "model.pt"
    
    save_paths = [] # store every possible path combination
    
    for model, method, dataset in product(model_names, pt_methods, dataset_names):
        save_path = os.path.join(base_path, model, method, dataset, weights_file_name)  
        
        # 创建目录  
        os.makedirs(save_path, exist_ok=True)  
        save_paths.append(save_path)  
    
    return save_paths



def fix_seed(seed):
    '''
    set the seed of the random number generator
    '''
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed) # CPU seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # GPU seed
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 



class GensimWord2VecWrapper:  
    def __init__(self,   
                 model: Union[str, Word2Vec, KeyedVectors],  
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):  
        """  
        初始化Word2Vec包装器  
        
        Args:  
            model: 可以是以下三种类型之一：  
                  - 预训练模型的路径（字符串）  
                  - 已加载的Word2Vec模型  
                  - 已加载的KeyedVectors模型  
            device: 运行设备，默认使用GPU如果可用  
        """  
        self.device = device  
        
        # 根据输入类型加载或设置模型  
        if isinstance(model, str):  
            # 如果是模型路径，则加载模型  
            try:  
                self.model = KeyedVectors.load(model)  
            except:  
                try:  
                    self.model = KeyedVectors.load_word2vec_format(model, binary=True)  
                except:  
                    self.model = KeyedVectors.load_word2vec_format(model, binary=False)  
        elif isinstance(model, (Word2Vec, KeyedVectors)):  
            # 如果是已加载的模型，直接使用  
            self.model = model.wv if isinstance(model, Word2Vec) else model  
        else:  
            raise ValueError("模型必须是路径字符串或Word2Vec/KeyedVectors实例")  
        
        self.vector_size = self.model.vector_size  
        print(f"模型加载完成。词向量维度: {self.vector_size}")  
    
    @classmethod  
    def train_new_model(cls,   
                       sentences: List[List[str]],   
                       vector_size: int = 100,  
                       window: int = 5,  
                       min_count: int = 1,  
                       workers: int = 4,  
                       sg: int = 0) -> 'GensimWord2VecWrapper':  
        """  
        训练新的Word2Vec模型  
        
        Args:  
            sentences: 训练语料，每个元素是一个词列表  
            vector_size: 词向量维度  
            window: 上下文窗口大小  
            min_count: 词的最小出现次数  
            workers: 训练的线程数  
            sg: 训练算法，0为CBOW（默认），1为Skip-gram  
            
        Returns:  
            GensimWord2VecWrapper实例  
        """  
        model = Word2Vec(sentences=sentences,  
                        vector_size=vector_size,  
                        window=window,  
                        min_count=min_count,  
                        workers=workers,  
                        sg=sg)  
        return cls(model)  

    def get_word_vector(self, word: str) -> torch.Tensor:  
        """  
        获取单个词的向量  
        
        Args:  
            word: 输入词  
            
        Returns:  
            torch.Tensor: 词向量  
        """  
        try:  
            vector = self.model[word]  
            return torch.tensor(vector, device=self.device)  
        except KeyError:  
            print(f"警告: 词'{word}'不在词表中，返回零向量")  
            return torch.zeros(self.vector_size, device=self.device)  

    def get_word_vectors(self,   
                        words: List[str],   
                        pad_to_length: Optional[int] = None) -> List[torch.Tensor]:  
        """  
        获取词列表的向量列表  
        
        Args:  
            words: 输入词列表  
            pad_to_length: 如果指定，将结果填充到指定长度  
            
        Returns:  
            List[torch.Tensor]: 词向量列表  
        """  
        vectors = [self.get_word_vector(word) for word in words]  
        
        if pad_to_length is not None:  
            # 填充到指定长度  
            padding = [torch.zeros(self.vector_size, device=self.device)   
                      for _ in range(pad_to_length - len(vectors))]  
            vectors.extend(padding)  
            vectors = vectors[:pad_to_length]  
            
        return vectors  

    def get_word_vectors_tensor(self,   
                              words: List[str],  
                              pad_to_length: Optional[int] = None) -> torch.Tensor:  
        """  
        获取词列表的向量张量（批处理形式）  
        
        Args:  
            words: 输入词列表  
            pad_to_length: 如果指定，将结果填充到指定长度  
            
        Returns:  
            torch.Tensor: 形状为(len(words), vector_size)的张量  
        """  
        vectors = self.get_word_vectors(words, pad_to_length)  
        return torch.stack(vectors)  
    

import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data.distributed import DistributedSampler 

def setup_distributed(use_cuda=True)->Tuple[torch.device, int]:
    """
    设置分布式训练环境
    Args:
        use_cuda: 是否使用CUDA设备
    Returns:
        device: 训练设备
        local_rank: 本地进程序号
    """
    if not use_cuda:
        return torch.device("cpu"), 0
    
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    # 检查进程组是否已经初始化  
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    return device, local_rank




def detect_param_grad_updates(model, epoch, step):
    '''
    检测可训练参数的梯度是否真的在更新
    '''  
    print(f"\n\n******************* Epoch:{epoch}, Step:{step}, Check Whether Gradient is Updating ***************8")
    for name, param in model.named_parameters():  
        if param.requires_grad:  
            print(f"{name}'s parameter mean: {param.data.mean().item() if param.data is not None else 'None'}")  
            print(f"{name}'s gradient norm: {param.grad.norm().item() if param.grad is not None else 'None'}") 
    
    print("\n\n================== NaN Gradient Detection ========================================")
    for name, param in model.named_parameters():  
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):  
            print(f"Gradient of {name} contains nan or inf at epoch:{epoch} step: {step}")
    print("**************************************************************\n\n\n")
    

def monitor_gradients(model, step):  
    print(f"\n\n********************** Monitor Gradients for step={step}******************************")
    grad_stats = {}  
    for name, param in model.named_parameters():  
        if param.requires_grad:
            if param.grad is not None:  
                grad_norm = param.grad.norm().item()  
                if grad_norm < 1e-8:  
                    print(f"Warning: Very small gradient for {name}: {grad_norm}") 
                grad_stats[name] = grad_norm
            else:
                print(f"Warning: No gradient (gradient is None) for {name}")
    
    if None in grad_stats.values() or grad_stats == {}:
        print(f"Warning: None in grad_stats.values() at step {step}")
        return
    
    # 检查梯度比例  
    max_grad = max(grad_stats.values())  
    min_grad = min(grad_stats.values()) 
    
    if min_grad == 0:
        print(f"Warning: min_grad is 0 at step {step}")
    elif max_grad / min_grad > 1000:  # 梯度比例阈值  
        print(f"Warning: Large gradient ratio 'max_grad/min_grad' at step {step}: {max_grad/min_grad}")
    
    print("*********************************************************************\n\n\n")


def record_epoch_param_state(model, param_monitor:Dict):
    # 添加参数状态监控  
    print("*************************** 记录上一轮训练后的参数 ********************************")
    for name, param in model.named_parameters():  
        if param.requires_grad:  
            if name not in param_monitor:  
                param_monitor[name] = []  
            param_monitor[name].append(param.data.clone().cpu())
    
    return param_monitor

def print_prediction_distribution(outputs,step,loss, num_options=2, logits=None, labels=None):
    '''
     logits: shape=(batch_size, num_options)
    '''
    print("*************************** 打印训练过程中的预测分布 ***************************************")
    with torch.no_grad():  
        preds = logits.argmax(dim=-1)  
        # print(f"\nStep {step} predictions distribution:",   
        #         torch.bincount(preds).cpu().numpy())  
        
        print(f"\nStep {step} Prediction Distribution:")  
        for i in range(num_options):  
            count = (preds == i).sum()  
            print(f"Option {i}: {count} ({count/len(preds):.2%})")  
            
        print("\n===============================================\n")
            
        print(f"\nStep {step} True Label Distribution:")  
        for i in range(num_options):  
            count = (labels == i).sum()  
            print(f"Option {i}: {count} ({count/len(labels):.2%})")  
            
        print("\n===============================================\n")
        print(f"Current loss: {loss.item():.4f}")  
    print("******************************************************************************************\n\n")

def check_param_after_epoch(model,param_monitor:Dict, epoch:int):
    print("************************ 检查每轮训练后的参数是否变化 ****************************")
    for name in param_monitor:  
        param_change = torch.norm(  
            # 当前轮-上一轮的差值
            param_monitor[name][-1] - param_monitor[name][-2]  
        ).item()  
        print(f"\nEpoch {epoch} {name} change: {param_change:.6f}")  

    print("*******************************************************************\n\n")



def get_model_path_by_model_name(model_name:str):
    '''
        根据模型名称获取模型路径
    '''
    path=None
    if "bert" in model_name.lower():
        path = Config['models'][model_name]['model_path']
    elif "SentenceTransformer" in model_name or "all-MiniLM-L6-v2" in model_name:
        path = Config['models'][model_name]['model_path'] 
    elif "qwen" in model_name.lower():
        path = Config['models']['qwen'][model_name]['model_path']
    elif "t5" in model_name.lower():
        path = Config['models']['t5'][model_name]['model_path']
    elif "llama" in model_name.lower():
        path = Config['models']['llama'][model_name]['model_path']
    elif "gpt2" in model_name.lower():
        path = Config['models']['gpt2'][model_name]['model_path']
    else:
        raise ValueError(f"model_name:{model_name} not supported, can not be found in `Config` dict")
    
    return path


def parse_training_arguments(config=None):
    parser = argparse.ArgumentParser(description="peft training")

    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="model used for training. Please select from [bert-base, bert-large, roberta-large, t5, qwen2.5]"
    )
    
    parser.add_argument(
        "--dataset_name", type=str, default="race", help="select a dataset from ['race','sciq','dream','commonsense_qa']"
    )
    
    parser.add_argument(
        "--classes_initiate_method", type=str, default="cluster", help="select from ['normal','lda','cluster']"
    )
    
    parser.add_argument(
        "--train_size", type=int, default=22000, help="size of the training set"
    )
    
    parser.add_argument(
        "--mixed_precision", type=str, default='no', help="whether to use mixed precision training"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size for training"
    )
    
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of epochs for training"
    )
    
    parser.add_argument(
        "--suffix_ratio", type=int, default=10, help="the ratio that suffix occupies in the max sequence length"
    )
    
    
    
    args = parser.parse_args()


    return args





class EarlyStopping:  
    def __init__(self, patience=5, threshold=1e-4):  
        self.patience = patience  
        self.threshold = threshold  
        self.counter = 0  
        self.best_loss = None  
        self.early_stop = False  
        
    def __call__(self, val_loss):  
        if self.best_loss is None:  
            self.best_loss = val_loss  
        elif val_loss > self.best_loss - self.threshold:  
            self.counter += 1  
            if self.counter >= self.patience:  
                self.early_stop = True  
        else:  
            self.best_loss = val_loss  
            self.counter = 0  
        
        return self.early_stop



def main():
    '''
    for testing
    '''
    
    





if __name__ == "__main__":
    # main()
    # make_save_dirs()
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    
    model, tokenizer = prepare_model_tokenizer(model_path,AutoModelForSequenceClassification, model_path)

    
    get_classifier_from_model(model)