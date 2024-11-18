
'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''
import os
import sys
from statistics import mean
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import default_data_collator
from sklearn.cluster import KMeans 
from sentence_transformers import SentenceTransformer 
from collections import defaultdict  

# 获取当前文件所在目录的父目录  
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

# 将父目录添加到sys.path  
sys.path.insert(0, parent_directory) 

from load import *

from config import Config

import openai
from openai import OpenAI
import os
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime

from dataclasses import dataclass

import sys  
import logging  
from datetime import datetime 

from typing import List, Tuple, Dict

client = OpenAI(
    
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url='https://api.feidaapi.com/v1'
)


@dataclass  
class Arguments:  
    """参数配置类"""  
    model: str = "gpt-4o"  # 或其他支持CoT的模型  
    method:str = "zero-shot-cot"
    max_length: int = 256  
    max_length_direct: int = 32  
    temperature: float = 0  
    direct_answer_trigger_for_zeroshot_cot: str = "Therefore, the answer (uppercase alphabetic letter) is"  
    num_samples: int = 1  
    log_dir: str = "./cot_log"  # 日志文件夹路径 
    api_time_interval:float = 1.0
    dataset_path: str = "../data/race"
    dataset: str = "race"



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
    torch.cuda.manual_seed_all(seed) # GPU seed
    torch.backends.cudnn.deterministic = True 


def decoder_for_gpt4(args, input, max_length):
    time.sleep(args.api_time_interval)
    
    
    if args.model == "gpt-4o":
        engine = "gpt-4o"
    
    elif args.model == "gpt-4o-mini":
        engine = "gpt-4o-mini"
    else:
        raise ValueError("GPT model is not properly defined ... please select gpt-4o or gpt-4o-mini")
    
    if ("few_shot" in args.method or "auto" in args.method): #  and engine == "code-davinci-002":
        response = client.chat.completions.create(
          model=engine,
        #   prompt=input,
          messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": input}
                    ],
          max_tokens=max_length,
          temperature=args.temperature,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )
    else:
        response = client.chat.completions.create( # zero-shot, zero-shot-cot
            model=engine,
            # prompt=input,
            messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": input}
                    ],
            max_tokens=max_length,
            temperature=args.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None # 生成文本遇到该字符时停止，None则不设置停止条件 
        )

    # return response["choices"][0]["text"]
    return response.choices[0].message.content




class Decoder():
    def __init__(self):
        pass
    
    def decode(self, args, input, max_length):
        response = decoder_for_gpt4(args, input, max_length)
        return response
    
    



def data_reader(args):
    pass









# Create dataset object before dataloader ...
class MyDataset(Dataset):
    pass



def get_reformated_dataset(args)->Tuple[Dataset, DatasetConfig]:
    '''
     获取数据集，并将数据集reformat成两列[question, answer]
    '''
    # questions = []
    # answers = []
    # decoder = json.JSONDecoder()
    

    dataset = None
    if args.dataset == 'race' or args.dataset == 'dream' or args.dataset == 'sciq' or args.dataset == 'commonsense_qa':
        # dataset_path = Config['datasets']['race']
        # dataset = load_dataset_from_huggingface(dataset_path, "all", split = 'train')


        dataset, config = preprocess_dataset_autocot(args.dataset)
        
        print(f"Dataset after loading: type:{type(dataset)}; features:{dataset.features}") 
        print(f"Dataset size after loading: {len(dataset) if dataset is not None else 'None'}")  
    
    else:
        raise ValueError("dataset is not properly defined ... Please select from [ race, dream, sciq, commonsense_qa]")

    return dataset, config


def setup_data_loader(args):
    '''
     return a huggingface dataset wrapped by a DataLoader
     
     it only contains 2 fields: {question, answer}
    
    '''
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        '''
        用于多个CPU进程之间的随机数种子同步
        '''
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " ,dataloader_num_workers)
    
    # load data and reformat the question and answer
    dataset, config = get_reformated_dataset(args)
    dataset_size = len(dataset)
    print(f"Dataset size = {dataset_size}") 
    if dataset_size == 0:  
        raise ValueError("Dataset is empty! Please check get_reformated_dataset function.") 

    
    # print("Dataset info:")  
    # print(f"Dataset type: {type(dataset)}")  
    # if hasattr(dataset, 'features'):  
    #     print(f"Dataset features: {dataset.features}")  
    
    # collate_fn可能返回了空字典
    def collate_fn(batch):
        '''
        必须配合 batch_size=1 一起使用！！！！！
        
        注意：
            如果batch_size>1, 那这么做实际上舍弃了其余的batch-1个样本！！！
        
        '''
        
        # 确保batch不为空  
        if not batch:  
            raise ValueError("Empty batch received") 
        # 返回单个样本，但确保包含所需的键  
        sample = batch[0]  
        if not (config.question_key in sample and config.label_key in sample):  
            raise KeyError(f"Required keys {config.question_key} and {config.label_key} not found in sample: {sample}")  
        
        return sample  
    
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=1,
        drop_last=False, #  如果最后一个批次的数据量小于 batch_size，是否丢弃该批次。
        num_workers=dataloader_num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,        
        collate_fn=collate_fn   # 这里不能用default_data_collator, 它专门用来处理tokenize后的数据
    )
    
    # default_data_collator期望处理的数据格式是：  
    # - 已经tokenize过的数据  
    # - 包含input_ids, attention_mask等字段  
    # - 或者是可以直接转换为tensor的数据 
    
    return dataloader, config



def get_text_embeddings(
    texts: List[str], 
    encoder: str = 'all-MiniLM-L6-v2',    
    batch_size: int = 32,  
    device: str = Config['device'],      
    )-> Tuple[np.ndarray, List[str]]:
    
    """  
    将DataLoader中的文本转换为向量表示  
    
    Args:  
    :params: texts: 一个数据集的question列表
    :params: encoder: SentenceTransformer模型名称  
    :params: batch_size: 批处理大小  
    :params: device: 设备类型 ('cuda', 'cpu' 或 None)  
    
    返回:  
    embeddings: 文本的向量表示  
    texts: 对应的原始文本列表  
    """  
    model = SentenceTransformer(encoder)  
    if device is not None:  
        model = model.to(device) 
        

    # 生成文本嵌入  
    embeddings = model.encode(  
        texts,  
        batch_size=batch_size,  
        show_progress_bar=True,  
        convert_to_numpy=True  
    )  
    
    return embeddings, texts

def cluster_dataloader(
    dataloader, 
    args, 
    config:DatasetConfig, 
    num_example = 300, 
    n_clusters=5, 
    random_state=42
    )->Tuple[np.ndarray, KMeans, Dict[int, List[str]]]:  
    """  
    对DataLoader中的前 num_example 个数据进行KMeans聚类  
    
    Args  
    :param: dataloader: torch.utils.data.DataLoader, 包含要聚类的数据  
    :param: n_clusters: int, 聚类的数量  
    :param: random_state: int, 随机种子
    :param: num_example 从数据集中取几个样本进行聚类
    
    返回:  
    labels: 聚类标签  
    kmeans: 训练好的KMeans模型  
    clusters: 每个簇的文本字典  
    """  
    
    # 收集DataLoader中的所有数据  
    texts = []  # List[str]
    steps = 0
    for batch in dataloader: 
        """batch
        {
            "question": ["What is the capital of France?","xxxxx", "yyyyy"]
            "answer": ["Paris", "xxx", "yyy"]
        }
        """ 
        # 如果batch是元组（比如同时包含数据和标签），只取第一个元素（数据）  
        # (data, label) = (tensor(batch_size, seq_len), tensor(batch_size,))
        if isinstance(batch, (tuple, list)):  
            batch = batch[0]  
        # 将数据移到CPU并转换为numpy数组
        questions = batch[config.question_key]  # List[str]
        for question in questions:
            if steps >= num_example:
                print("Reach the max number {} of examples, stop collecting questions...".format(num_example))
                break
            steps+=1
            texts.append(question)  
    
    
    embeddings, texts= get_text_embeddings(texts, args.encoder)
    
    
    # 使用KMeans进行聚类  
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)  
    labels = kmeans.fit_predict(embeddings) 
    
    assert len(labels)==len(embeddings), "length of embeddings and labels are not the same"
    assert len(labels)==len(texts), "length of labels and texts are not the same" 
    
    
    # 计算每个样本到其质心的距离 
    distances = np.zeros(len(texts))   # index = sentence_id, value=distance
    for i, (embedding, label) in enumerate(zip(embeddings, labels)): 
        centroid = kmeans.cluster_centers_[label]  
        distance = np.linalg(embedding-centroid) 
        distance[i] = distance

    # 将文本按簇组织  
    clusters = defaultdict(list)
    for text, label, distance in zip(texts, labels, distances): 
        clusters[label].append((text, distance))
    
    # 对每个簇内的文本按距离排序  
    sorted_clusters = {}  
    for label in clusters:  
        # 按距离从近到远排序  
        sorted_texts = sorted(clusters[label], key=lambda x: x[1])  
        sorted_clusters[label] = sorted_texts  
    
    return labels, kmeans, sorted_clusters 


def get_k_questions_from_clusters(
    clusters: Dict[int, List[str]],
    k: int = 10
    )->List[str]:
    '''
    从聚类结果中随机选择k个问题
    '''
    pass
    
def answer_cleansing(args, pred, must_choice=False):
    '''
    历史遗留函数， race数据集用不了规则匹配，因为他无规则
    
    '''

    print("prediction length before cleaning : ",len(pred))
    
    print("some of the prediction are: \n", pred[:20])


    if args.method == "few_shot":
        # split 之前： "Therefore, among A to D, the answer is A."
        preds = pred.split(args.direct_answer_trigger_for_fewshot) # trigger: "The answer is:"
        # True, 表示模型输出的结果中包含了多个候选答案
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1] 
    if args.method == "zero_shot_cot":
        # split 之前： "Therefore, among A to D, the answer is A."
        preds = pred.split(args.direct_answer_trigger_for_zeroshot_cot) # trigger: "The answer is:"
        pred = preds[-1]
    
    if args.method == "auto_cot":
        pattern = r'(?i)(?=[Oo]ption\s*:?\s*[A-Z])'  
        segments:list = re.split(pattern, pred) 
        segments.pop(0)
        pred = segments[0][1]
        
    # 正则表达式匹配答案
    if args.dataset == "race":
        pred:List = re.findall(r'A|B|C|D', pred) # ['2', '3', '4', '5']
    elif args.dataset == "dream":
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset == "sciq":
        pred = re.findall(r'A|B|C|D', pred)
    elif args.dataset == "commensense_qa":
        pred = re.findall(r'A|B|C|D|E', pred)
    else:
        raise ValueError("dataset is not properly defined ... Please select from [race, sciq, dream, commonsense_qa]")
    
    
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot", "auto_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ... you should select from [ few_shot, few_shot_cot, zero_shot, zero_shot_cot, auto_cot ]")

    
    # for arithmetic tasks, if a word ends with period, it will be omitted ...
    # 数学答案不能包含句号
    if pred != "": # assume we get '78.9.'
        if pred[-1] == ".":
            pred = pred[:-1]

    print("prediction after cleaning : " + pred)
    
            
    return pred



def create_demo_text(args, cot_flag)->str:
    '''
     将demo文件中的json对象数组拼接成一整个字符串返回
     
     适用于 few-shot， few-shot-cot, auto-cot
     
     :return: cot_flag: True 表示拼接cot格式的demos（如auto-cot)， False 表示不使用cot格式拼接
    '''
    x, z, y = [], [], []
    
    # demo文件中只包含了 k 个 clusters 对应的k个demos
    
    with open(args.demo_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])
    
    index_list = list(range(len(x)))
    
    demo_text =""
    
    for i in index_list:
        if cot_flag:
            demo_text += x[i] + " " + z[i] + " " + \
                         args.direct_answer_trigger_for_zeroshot_cot + " " + y[i] + ".\n\n"
        else:
            # args.direct_answer_trigger_for_fewshot： "The answer is:"
            demo_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text
    
    
    
    
    
    


 
class LoggerWriter:  
    """
    A class that:
        reorients the output of the 'print' to a logger
    """  
    def __init__(self, logger, level):  
        self.logger = logger  
        self.level = level  

    def write(self, message):  
        # check whether message is not None or empty, and message should not only contain a space
        if message and not message.isspace():  
            self.logger.log(self.level, message)  

    def flush(self):  
        pass  
    

def setup_logger(dataset_name: str, args) -> logging.Logger:  
    """  
    set up a logger that can only by used by zero-shot-cot  
    Args:  
        dataset_name: 数据集名称  
        args: args produced by parsearg
    Returns:  
        配置好的logger  
    """  
    # 创建log目录（如果不存在）  
    if not os.path.exists(args.log_dir):  
        os.makedirs(args.log_dir)  

    # 设置日志文件路径  
    log_file = os.path.join(args.log_dir, f"{dataset_name}_zero_shot_cot.log")  
    
    # 配置logger  
    logger = logging.getLogger('cot_logger')  
    logger.setLevel(logging.INFO)  
    
    # 清除之前的handlers（如果有）  
    if logger.handlers:  
        logger.handlers.clear()  
    
    # 创建控制台处理器  
    console_handler = logging.StreamHandler()  
    console_handler.setLevel(logging.INFO)  
    
    # 添加文件handler  
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')  
    file_handler.setLevel(logging.INFO)  
    
    # 设置日志格式      
    # %(message)s 表示只记录日志消息本身，不包含其他额外的信息，如时间戳、日志级别等。
    formatter = logging.Formatter('%(message)s')  
    file_handler.setFormatter(formatter)  
    console_handler.setFormatter(formatter)  
    
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler) 
    
    return logger  






def extract_answer(text: str) -> str:
    """
    从生成的文本中提取多项选择题的答案（大写字母）
    
    Args:
        text: 生成的文本，包含类似 "The answer is A" 的片段
        
    Returns:
        str: 提取出的大写字母答案。如果没有找到有效答案，返回空字符串
        
    Examples:
        >>> extract_answer("Let's think about this... The answer is A")
        'A'
        >>> extract_answer("After careful consideration, The answer is C.")
        'C'
        >>> extract_answer("The answer is B because...")
        'B'
        >>> extract_answer("No clear answer")
        ''
    """
    # 使用正则表达式匹配 "The answer is X" 模式，其中X是一个大写字母
    pattern = r"the answer is ([A-D])[.\s]" # [.\s] 匹配一个句号或空格
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)  # 如果找到了匹配，就会返回匹配的第一个捕获组, 返回匹配到的大写字母
    
    # 如果上面的模式没有匹配到，尝试匹配文本中最后出现的单个大写字母答案、
    # (?: xxxx) 非捕获组，只匹配，不捕获，一般用来做 (?:ab)+ 这样的叠加或叠乘操作
    pattern_backup = r"([A-D])(?:\.|$|\s)"
    matches = list(re.finditer(pattern_backup, text))
    if matches:
        return matches[-1].group(1)  # 返回最后一个匹配到的大写字母
        
    return ""  # 如果没有找到任何有效答案，返回空字符串







if __name__ == "__main__":
    args = {
        "dataset": "race",
        "random_seed": 42,
        "max_num_worker": 5,
        "minibatch_size": 1,
    }
    from collections import namedtuple  
    MyDict = namedtuple('MyDict', args.keys())  
    args = MyDict(**args)  
    # ds = get_reformated_dataset(args)
    
    dataloader = setup_data_loader(args)