
'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

from statistics import mean
from torch.utils.data import (
    Dataset,
    DataLoader
)
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

from typing import List, Tuple, Dict

client = OpenAI(
    
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url='https://api.feidaapi.com/v1'
)



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
          stop=["\n"]
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
    return response.choices[0].message['content']




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



def get_reformated_dataset(args):
    '''
     获取数据集，并将数据集reformat成两列[question, answer]
    '''
    pass


def setup_data_loader(args):
    '''
     return a huggingface dataset wrapped by a DataLoader
    
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
    dataset = MyDataset(args)
    
    def collate_fn(batch):
        return batch[0]
    
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.minibatch_size,
        drop_last=False,
        num_workers=dataloader_num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=collate_fn # 需确保batch_size = 1
    )
    
    return dataloader
    
def answer_cleansing(args, pred, must_choice=False):

    print("prediction value before cleaning : " + pred)


    if args.method in ("few_shot", "few_shot_cot", "auto_cot"):
        # split 之前： "The answer is: 1 or The answer is: 2 or 3 or 4 or 5"
        # preds after split = ["1 or", "2 or 3 or 4 or 5" ]
        preds = pred.split(args.direct_answer_trigger_for_fewshot) # trigger: "The answer is:"
        # True, 表示模型输出的结果中包含了多个候选答案
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1] # 挑最后一个候选答案  "2 or 3 or 4 or 5"

    
    if args.dataset == "race":
        pred:List = re.findall(r'A|B|C|D|E', pred) # ['2', '3', '4', '5']
    elif args.dataset == "record":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset == "multirc":
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset == "arc":
        pred = re.findall(r'A|B|C', pred)
    else:
        raise ValueError("dataset is not properly defined ... Please select from [race, record, multirc, arc]")
    
    
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
    '''
    x, z, y = [], [], []
    
    
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
    
    