import os
import torch
import logging
import evaluate
import numpy as np
from itertools import product
from config import Config


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
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
)

from config import Config

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support



def get_model_name_using_model(model):
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
        if config.hidden_size == 4096:  
            return "qwen-14b-chat"  
        elif config.hidden_size == 5120:  
            return "qwen-16b-chat"  
    else:  
        # 无法匹配已知模型，返回未知模型提示  
        return "unknown model, please check your config, it should be [bert | llama | qwen2]"  


def get_vocab_embeddings_from_model(model, token_ids:torch.LongTensor):
    '''
     model is a pretrained model
     
     token_ids: a tensor of shape (num_tokens, 1)
     
     return a tensor of shape (num_tokens, 1, hidden_size)
    '''
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
    else:  
        raise AttributeError(f"Can not find the embedding layer in the model. Please check the model type {type(model).__name__}.") 




def prepare_model_tokenizer(model_path, auto_model_class = AutoModel, tokenizer_path = None):
    '''
     return model, tokenizer
    '''
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, num_labels=4)
    
    model = auto_model_class.from_pretrained(model_path, config=config)
    
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    print(f"Model's current num_labels: {model.config.num_labels}") 
     
    model_config = model.config
    model_name_or_path = model_config.name_or_path
    print("model_name_or_path = ", model_name_or_path)
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    print("padding_side = ", padding_side)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    
    return model, tokenizer








def get_logger(name="my_logger", logging_dir = None,log_level="INFO"):
    # 配置基本日志设置：设置日志的最低显示级别和格式  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  

    # 创建logger对象  
    logger = logging.getLogger(name)  

    # 定义日志文件的路径  
    log_file_path = logging_dir  

    # 创建一个FileHandler处理对象，将日志输出到文件  
    file_handler = logging.FileHandler(log_file_path)  
    # 设置文件处理器的日志级别  
    file_handler.setLevel(logging.INFO)  

    # 创建格式化器，将格式对象与处理器关联  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    file_handler.setFormatter(formatter)  

    # 将FileHandler添加到logger中  
    logger.addHandler(file_handler)  






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
    
    
    
    


def main():
    '''
    for testing
    '''
    
    





if __name__ == "__main__":
    # main()
    make_save_dirs()