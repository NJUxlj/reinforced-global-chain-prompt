import pandas as pd  
from datasets import (
    Dataset, 
    DatasetDict,
    DatasetBuilder, 
    load_dataset, 
    load_dataset_builder,
    concatenate_datasets,
)
from dataclasses import dataclass 
from transformers import (
    AutoTokenizer, 
    BertTokenizerFast,
    AutoModel,
    PreTrainedTokenizer,  
    BatchEncoding,  
)
import random
import torch
import json
import os
import re

from config import Config
from config import NUM_PROCESSES, NUM_CPU_PROCESSES

from typing import List, Dict, Tuple ,Union, Any, Optional



# 初始化分词器  
tokenizer = AutoTokenizer.from_pretrained(Config["models"]["bert-base-uncased"]["model_path"])  


@dataclass  
class DatasetConfig:  
    """数据集配置类"""  
    name: str  
    article_key: str  
    question_key: str  
    options_key: str  
    label_key: str  
    local_path: Optional[str] = None  # 添加本地路径  
    file_format: str = "huggingface"  # 文件格式：'huggingface' 或 'json'  
    subset: str = None  


def load_dataset_from_huggingface(dataset_path, subset_name = None, split = None, cache_dir = None):
    '''
    load dataset from huggingface hub
    
        dataset_path: "/your/path/glue"
        
        subset_name: "sst2"
        
        split: "train", "validation"
        
        
        dataset = load_dataset("super_glue", "multirc", split="train") 
    '''
    
    if subset_name:
        ds = load_dataset(dataset_path, subset_name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_path, split=split, cache_dir=cache_dir)

    return ds




def preprocess_function_race_pt(examples, text_column = "article", label_column  ="answer", dataset_name = 'race', max_length = 492):
    
    
    
    ''' 
      iteratively modify every batch of data
      
      Args:
          text_column: the column name of the text
          
          ds_builder: use it to get the name of dataset
          
          classes: the set of labels (str) e.g. ['A', 'B', 'C', 'D']    ['good', 'bad']
          
    '''
    batch_size = len(examples[text_column])
    
    # use hard prompt to combine [article, question, options] together
    # race's fields are [article, question, options, answer]
    inputs = [f"Artical:{examples['article'][index]}\n\nQuestion:{examples['question'][index]}\n\n \
              Options:{examples['options'][index]}\n\nAnswer:" for index, x in enumerate(examples[text_column])]  

    # targets = [str(x) for x in examples['answer']] # shape = (batch_size, )
    
    global tokenizer
    
    # return a dict with keys = ['input_ids', 'attention_mask',...]
    # if no return_tensors = 'pt', return list as values
    model_inputs = tokenizer(inputs)
    # labels = tokenizer(targets) 
    
    labels = []
    for answer in examples['answer']:
        labels.append(ord(answer)-ord('A'))
    
    # labels['input_ids'].shape = (batch_size,  1)  暂定是该形状
    
    # classes = list(set(targets))
    classes = Config['classes']['race']    
    for i in range(batch_size):
        sample_input_ids = model_inputs['input_ids'][i] # shape = (seq_len)
        
        # 加上[CLS]
        sample_input_ids += [tokenizer.cls_token_id]
        # label_input_ids = labels['input_ids'][i]
        label_input_ids = labels[i]
        
        
        # padding
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id]*(max_length-len(sample_input_ids))+sample_input_ids
        
        # model_inputs["input_ids"].shape = (batch_size, max_length)
        # model_inputs["input_ids"][i].shape = (max_length)
        model_inputs['attention_mask'][i] = [0]*(max_length-len(sample_input_ids))+[1]+model_inputs['attention_mask'][i]
        
        # labels[i] 在分类任务中不用补齐
        # labels["input_ids"][i] = [-100]*(max_length - len(label_input_ids))+label_input_ids
        
        # truncate and transfer to tensor
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        # labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        # labels[i] = torch.LongTensor(label_input_ids)  
        
    # model_inputs['labels'] = labels["input_ids"]
    model_inputs['labels'] = labels
    
    
    # 打印形状
    print("batch_size = ", len(model_inputs["input_ids"]), " or ", len(model_inputs["labels"]))
    print("seq_length for input = ", len(model_inputs['input_ids'][1]))
    # print("seq_length for label = ", len(model_inputs["labels"][1]))
    print("seq_length for attention_mask = ", len(model_inputs["attention_mask"][1]))
    
    print("=============================================================================")
    # print("labels = \n", model_inputs["labels"])
    print("model_inputs[\"labels\"][1]) = ", model_inputs["labels"][1])
        
    return model_inputs
    

def preprocess_function_race(examples, first_four_columns = ["article", "question", "options", "answer"],
                             dataset_name = 'race', max_length = 512, tokenizer = None)->Dict[str,Union[List,List[List]]]:
    
    ''' 
      iteratively modify every batch of data
      
      Args:
          classes: the set of labels (str) e.g. ['A', 'B', 'C', 'D']    ['good', 'bad']
        
        return: 

                {  
                    'input_ids': [[101, ..., 102], [101, ..., 102], ...],          # 每个子列表对应一个样本的 token IDs  
                    'attention_mask': [[1, ..., 1], [1, ..., 0], ...],             # 1 表示实际 token，0 表示填充  
                    'labels': [0, 2, 3, ...]                                        # 标签的整数索引表示  
                }  
          
    '''
    assert tokenizer is not None, "tokenizer in \"preprocess_function_race\" is None, please assign one"
    
    batch_size = len(examples[first_four_columns[0]])
    
    
    # # 构建输入文本  
    # inputs = [  
    #     f"Article: {examples['article'][i]}\n\nQuestion: {examples['question'][i]}\n\nOptions: {examples['options'][i]}\n\nAnswer:"  
    #     for i in range(len(examples[text_column]))  
    # ]  
    
    # 初始化结果列表  
    input_texts = []  
    labels = []  
    
    # 处理每个样本  
    for i in range(batch_size):  
        # 获取当前样本的各个字段  
        article = examples[first_four_columns[0]][i]  # article/support  
        question = examples[first_four_columns[1]][i]  
        options = examples[first_four_columns[2]][i]  # 已经带有A/B/C/D标签的选项列表  
        answer = examples[first_four_columns[3]][i]   # 答案标签（A/B/C/D）  
        
        # 将选项转换为字典格式，方便后续处理  
        # options_dict = {opt.split(". ")[0]: opt.split(". ")[1] for opt in options}  
        
        # 构建输入文本  
        # 格式：[CLS] 文章 [SEP] 问题 [SEP] 选项A [SEP] 选项B [SEP] 选项C [SEP] 选项D [SEP]  
        input_text = f'''
                        Article:{article}
                        Question:{question}
                        Options:
                        {options[0]}
                        {options[1]}
                        {options[2]}
                        {options[3]} 
                        Answer:
                        '''
        
        # 将答案标签转换为数字（A->0, B->1, C->2, D->3）  
        label = ord(answer) - ord('A')  
        
        input_texts.append(input_text)  
        labels.append(label)  
    
    model_inputs = tokenizer(  
        input_texts,  
        padding = "max_length",
        truncation=True,  
        max_length=max_length,  
        add_special_tokens=True,  # 确保添加特殊标记  [CLS] [SEP]
    )  
    
    # 确保标签符合n_classes，不超范围  
    assert all(0 <= label < 4 for label in labels), "There are labels out of range [0, 3]." 

    # print("labels = ", labels)
    # labels = torch.tensor(labels, dtype=torch.long)
    model_inputs['labels'] = labels  # 保持为整数列表  

    
    return model_inputs  


def preprocess_function_multirc(examples, text_column = "article", label_column  ="answer", 
                                dataset_name = 'multirc', max_length = 512)->Dict[str,Union[List,List[List]]]:
    """ 
    处理multirc数据集的预处理函数，将问题和选项合并为一个句子，并添加特殊标记。
    
    确保 dataset = load_dataset("super_glue", "multirc", split="train")  
    """
    


def preprocess_function_arc(examples, text_column = "article", label_column  ="answer", 
                            dataset_name = 'arc', max_length = 512)->Dict[str,Union[List,List[List]]]:
    pass



def preprocess_function_sciq(examples, first_four_columns = ["article", "question", "options", "answer"], 
                               dataset_name = 'sciq', max_length = 512, tokenizer = None)->Dict[str,Union[List,List[List]]]:
    """  
    预处理SciQ数据集的样本，准备用于模型输入  
    
    Args:  
        examples: 包含批量样本的字典  
        first_four_columns: 数据集的前四个关键列名  
        dataset_name: 数据集名称  
        max_length: 最大序列长度  
        
    Returns:  
        处理后的字典，包含以下字段：  
        - input_ids: 输入序列的token IDs  
        - attention_mask: 注意力掩码  
        - token_type_ids: token类型IDs（如果使用BERT类模型）  
        - labels: 标签  
    """  
    
    assert tokenizer is not None, "tokenizer in \"preprocess_function_sciq\" is None, please assign one"
    
    # 获取批次大小  
    batch_size = len(examples[first_four_columns[0]])  
    
    # 初始化结果列表  
    input_texts = []  
    labels = []  
    
    # 处理每个样本  
    for i in range(batch_size):  
        # 获取当前样本的各个字段  
        support = examples[first_four_columns[0]][i]  # article/support  
        question = examples[first_four_columns[1]][i]  
        options = examples[first_four_columns[2]][i]  # 已经带有A/B/C/D标签的选项列表  
        answer = examples[first_four_columns[3]][i]   # 答案标签（A/B/C/D）  
        
        # 将选项转换为字典格式，方便后续处理  
        # options_dict = {opt.split(". ")[0]: opt.split(". ")[1] for opt in options}  
        
        # 构建输入文本  
        # 格式：[CLS] 文章 [SEP] 问题 [SEP] 选项A [SEP] 选项B [SEP] 选项C [SEP] 选项D [SEP]  
        input_text = f'''
                        Context:{support}
                        Question:{question}
                        Options:
                        {options[0]}
                        {options[1]}
                        {options[2]}
                        {options[3]} 
                        Answer:
                        '''
        
        # 将答案标签转换为数字（A->0, B->1, C->2, D->3）  
        label = ord(answer) - ord('A')  
        
        input_texts.append(input_text)  
        labels.append(label)  
    
    # 使用tokenizer处理文本（假设tokenizer是类的成员变量）  
    model_inputs = tokenizer(  
        input_texts,  
        padding="max_length",  
        truncation=True,  
        max_length=max_length,  
        # return_tensors=None  # 返回列表而不是张量  
        add_special_tokens=True  # 确保添加特殊标记  [CLS] [SEP]
    )  
    
    # 添加标签到结果中  
    model_inputs["labels"] = labels  
    
    return model_inputs  

def preprocess_function_dream(examples, first_four_columns = ["article", "question", "options", "answer"],
                               dataset_name = 'dream', max_length = 512, tokenizer = None)->Dict[str,Union[List,List[List]]]:
    """  
    预处理 dream 数据集的样本，准备用于模型输入  
    
    Args:  
        examples: 包含批量样本的字典  
        first_four_columns: 数据集的前四个关键列名  
        dataset_name: 数据集名称  
        max_length: 最大序列长度  
        
    Returns:  
        处理后的字典，包含以下字段：  
        - input_ids: 输入序列的token IDs  
        - attention_mask: 注意力掩码  
        - token_type_ids: token类型IDs（如果使用BERT类模型）  
        - labels: 标签  
    """  
    
    assert tokenizer is not None, "tokenizer in \"preprocess_function_sciq\" is None, please assign one"
    
    # 获取批次大小  
    batch_size = len(examples[first_four_columns[0]])  
    
    # 初始化结果列表  
    input_texts = []  
    labels = []  
    
    # 处理每个样本  
    for i in range(batch_size):  
        # 获取当前样本的各个字段  
        dialogue = examples[first_four_columns[0]][i]  # article/support  
        question = examples[first_four_columns[1]][i]  
        options = examples[first_four_columns[2]][i]  # 已经带有A/B/C/D标签的选项列表  
        answer = examples[first_four_columns[3]][i]   # 答案标签（A/B/C/D）  
        
        # 将选项转换为字典格式，方便后续处理  
        # options_dict = {opt.split(". ")[0]: opt.split(". ")[1] for opt in options}  
        
        # 构建输入文本  
        # 格式：[CLS] 文章 [SEP] 问题 [SEP] 选项A [SEP] 选项B [SEP] 选项C [SEP] 选项D [SEP]  
        input_text = f'''
                        Dialogue:{dialogue}
                        Question:{question}
                        Options:
                        {options[0]}
                        {options[1]}
                        {options[2]}
                        Answer:
                        '''
        
        # 将答案标签转换为数字（A->0, B->1, C->2, D->3）  
        label = ord(answer) - ord('A')  
        
        input_texts.append(input_text)  
        labels.append(label)  
    
    # 使用tokenizer处理文本（假设tokenizer是类的成员变量）  
    model_inputs = tokenizer(  
        input_texts,  
        padding="max_length",  
        truncation=True,  
        max_length=max_length,  
        # return_tensors=None  # 返回列表而不是张量  
        add_special_tokens=True,  # 确保添加特殊标记  [CLS] [SEP]
        # batch_size = 32,
    )  
    
    # 添加标签到结果中  
    model_inputs["labels"] = labels  
    
    return model_inputs 


def preprocess_function_commonsense_qa(examples, first_four_columns = ["article", "question", "options", "answer"], 
                               dataset_name = 'commonsense_qa', max_length = 512, tokenizer = None)->Dict[str,Union[List,List[List]]]:
    """  
    预处理commonsense_qa数据集的样本，准备用于模型输入  
    
    Args:  
        examples: 包含批量样本的字典  
        first_four_columns: 数据集的前四个关键列名  
        dataset_name: 数据集名称  
        max_length: 最大序列长度  
        
    Returns:  
        处理后的字典，包含以下字段：  
        - input_ids: 输入序列的token IDs  
        - attention_mask: 注意力掩码  
        - token_type_ids: token类型IDs（如果使用BERT类模型）  
        - labels: 标签  
    """  
    
    assert tokenizer is not None, "tokenizer in \"preprocess_function_commonsense_qa\" is None, please assign one"
    
    # 获取批次大小  
    batch_size = len(examples[first_four_columns[0]])  
    
    # 初始化结果列表  
    input_texts = []  
    labels = []  
    
    # print(" first_four_columns = ", first_four_columns)
    print()
    # print("examples = \n", examples)
    
    # 处理每个样本  
    for i in range(batch_size):  
        # 获取当前样本的各个字段  
        question_concept = examples[first_four_columns[0]][i]  # article/support  
        question = examples[first_four_columns[1]][i]  
        options = examples[first_four_columns[2]][i]  # 已经带有A/B/C/D标签的选项列表  
        answer = examples[first_four_columns[3]][i]   # 答案标签（A/B/C/D）  
        
        # print("question_concept =", question_concept)
        # print("question =", question)
        # print("options =", options)
        # print("answer =", answer)  
             
        input_text = f'''
                        Question concept:{question_concept}
                        Question:{question}
                        Options:
                        {options[0]}
                        {options[1]}
                        {options[2]}
                        {options[3]} 
                        Answer:
                        '''
        
        # 将答案标签转换为数字（A->0, B->1, C->2, D->3）  
        label = ord(answer) - ord('A')  
        
        input_texts.append(input_text)  
        labels.append(label)  
    
    # 使用tokenizer处理文本（假设tokenizer是类的成员变量）  
    model_inputs = tokenizer(  
        input_texts,  
        padding="max_length",  
        truncation=True,  
        max_length=max_length,  
        # return_tensors=None  # 返回列表而不是张量  
        add_special_tokens=True,  # 确保添加特殊标记  [CLS] [SEP]
    )  
    
    # 添加标签到结果中  
    model_inputs["labels"] = labels  
    
    return model_inputs




    



def choose_dataset(dataset_name:str, split = None)->Dataset:
    '''
    最顶层的数据集获取封装
    
     return a huggingface dataset 

     Args:
        dataset_name: str, dataset name, choose from [race, arc, multirc, record]
        split: str, split of dataset, choose from [train, valid, test]
    Returns:
        ds: Dataset, a huggingface dataset
    '''
    dataset_wrapper = McqDatasetWrapper()
    ds, _ = dataset_wrapper.load_mcq_dataset(dataset_name, split)
    # ds = None
    # if dataset_name == "race":
    #     dataset_path = Config["datasets"][dataset_name]
    #     ds = load_dataset_from_huggingface(dataset_path, "high", split=split)
    # elif dataset_name == "arc":
    #     dataset_path = Config["datasets"][dataset_name]
    #     ds = load_dataset_from_huggingface(dataset_path, "multiple_choice", split=split)
    # elif dataset_name == "multirc":
    #     cache_dir = Config["datasets"]["super_glue"]
    #     ds = load_dataset_from_huggingface("super_glue", "multirc", split=split, cache_dir=cache_dir)
    # elif dataset_name == "record":
    #     cache_dir = Config["datasets"]["super_glue"]
    #     ds = load_dataset_from_huggingface("super_glue", "record", split=split, cache_dir=cache_dir)
    # else:
    #     raise ValueError(f"Unsupported dataset name: {dataset_name}, please select from [race, arc, multirc, record]")
    
  
    return ds


def preprocess_race(ds: Dataset, tokenizer:AutoTokenizer):
    '''
        preprocess race dataset in coarse-grained manner
        
        model_config: model.config
        
        return ds, classes, tokenizer
    '''
    # 
    print(ds["train"].features['answer'])
    # 提取训练集的标签列  
    train_labels = ds['train']['answer']  

    # 获取标签的唯一值集合  
    classes = sorted(list(set(train_labels)))
    # classes = [k.strip() for k in ds["train"].features["answer"].names]
    
    print("classes = ", classes)
    print("num_classes = ", len(classes))
    
    # race的Answer 就是 A, B, C, D. no need for mapping
    ds = ds.map(
        lambda x: x,
        batched=True,
        num_proc=1,
    )

    

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
    # get the max length of the label's input_ids
    target_max_length = max([len(tokenizer(class_label, max_length =1, padding = "max_length", truncation=True)["input_ids"]) for class_label in classes])
    print("target_max_length = ", target_max_length)
    
    print(f"dataset \"race\" is ready to be used ~~~")
    return ds, classes, tokenizer





def preprocess_func_autocot(config: DatasetConfig, examples:Dict[str,List]):
    '''
    预处理函数：将article、question和options字段合并成新的question字段  
                
                最后仅保留question和answer字段，remove 掉article和options字段
            no tokenization !
    Args:  
        examples:Dict[ str, List ]: 数据集中的一个批次样本  
        dataset_name: 数据集名称，用于选择预处理函数  
    Returns:  
        处理后的样本     
    '''
    dataset_name =config.name
    

    contexts = [str(art) for art in examples[config.article_key]]  
    questions = [str(q) for q in examples[config.question_key]]  
    options = [str(opt) for opt in examples[config.options_key]] 
    
    new_questions: List[str] = [  
        f'''{config.article_key}:
            {context}
            {config.question_key}:
            {question}
            {config.options_key}:
            {option}
        ''' 
        
         
        for context, question, option in zip(contexts, questions, options)  
    ]  
    
    examples[config.question_key] = new_questions 
    
    examples.pop(config.article_key)  
    examples.pop(config.options_key)  
    return examples




def preprocess_dataset_autocot(dataset_name):
    """  
    处理整个数据集  [必须是训练集]
    
    Args:  
        dataset: 原始的训练集 (split = train)  
        dataset_name: 数据集名称，用于选择预处理函数
    Returns:  
        处理后的数据集  
    """ 
    wrapper = McqDatasetWrapper()
    dataset, first_four_columns = wrapper.load_mcq_dataset(dataset_name)
    config = wrapper.dataset_configs[dataset_name]
    
    train_ds = dataset['train']
    
    # 先预处理第一个样本试试看 
    print("\nTesting preprocess_func_autocot with first example...")  
    try:  
        first_example = train_ds[0]  
        print(f"First example before processing: {first_example}")  
        # processed_example = preprocess_func_autocot(config, {'examples': [first_example]})  
        processed_example = preprocess_func_autocot(config, first_example)  
        
        print(f"First example after processing: {processed_example}")  
    except Exception as e:  
        print(f"Error in test first example: {str(e)}")  
        import traceback  
        print(f"Traceback:\n{traceback.format_exc()}")  
    
    print("\nFirst example testing succeed, now, Starting dataset mapping...")  
    processed_dataset = train_ds.map(
        function= lambda examples: preprocess_func_autocot(config, examples),
        batched=True,
        batch_size= Config['batch_size'],
        num_proc=NUM_CPU_PROCESSES,
        # remove_columns= ['example_id'],           # dataset.column_names,
        load_from_cache_file=False,
        desc=f"Running tokenizer on dataset {dataset_name}, when doing Auto-CoT",
    )

    print(f"\nProcessed dataset type: {type(processed_dataset)}")
    print(f"Processed dataset name: {dataset_name}")  
    print(f"Processed dataset size: {len(processed_dataset)}")  
    if hasattr(processed_dataset, 'column_names'):  
        print(f"Processed dataset columns: {processed_dataset.column_names}")  
        
    print("Processed dataset example[0]:\n\n", processed_dataset[0])

    return processed_dataset









########## Load and Reformat all datasets for PEFT tasks ###############






class McqDatasetWrapper:  
    '''
     This class provides methods for loading multiple choice datasets inin unified "Dataset" type
     
     It can handle different different format {"huggingface", "json"}
    '''
    def __init__(  
        self,  
        model_name_or_path: str = Config['models']['bert-base-uncased']['model_path'],  
        max_seq_length: int = 512,  
        label_map: Dict[str, int] = None,
        split = None  
    ):  
        """  
        初始化预处理器  
        
        Args:  
            model_name_or_path: BERT模型名称  
            max_seq_length: 最大序列长度  
            label_map: 标签映射字典，例如 {"A": 0, "B": 1, "C": 2, "D": 3}  
        """  
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)  
        self.max_seq_length = max_seq_length  
        self.label_map = label_map or {"A": 0, "B": 1, "C": 2, "D": 3}  
        self.split = None
        # 定义各个数据集的配置  
        self.dataset_configs = {  
            "race": DatasetConfig(  
                name="race",  
                article_key="article",  
                question_key="question",  
                options_key="options",  
                label_key="answer",  
                local_path=Config['datasets']['race'],  
                file_format="huggingface",
                subset="all",
            ),  
            "arc": DatasetConfig(  
                name="arc",  
                article_key="context",  
                question_key="question",  
                options_key="choices",  
                label_key="answerKey",  
                local_path=Config['datasets']['race'],  
                file_format="huggingface",  
                subset="ARC-Challenge"  
            ),  
            "multirc": DatasetConfig(  
                name="multirc",  
                article_key="paragraph",  
                question_key="question",  
                options_key="answer",  
                label_key="label",  
                local_path=Config['datasets']['multirc']['all'],  
                file_format="json"  
            ),  
            "dream": DatasetConfig(  
                name="dream",  
                article_key="dialogue",  
                question_key="question",  
                options_key="options",  
                label_key="label",  
                local_path=Config['datasets']['dream']['all'],  
                file_format="json"  
            ),
            "sciq": DatasetConfig(  
                name="sciq",  
                article_key="support",  
                question_key="question",  
                options_key="options",  
                label_key="answer",  
                local_path=Config['datasets']['sciq'],  
                file_format="huggingface"  
            ),
            "commonsense_qa": DatasetConfig(  
                name="commonsense_qa",  
                article_key="question_concept",  
                question_key="question",  
                options_key="options",  
                label_key="answer",  
                local_path=Config['datasets']['commonsense_qa'],  
                file_format="huggingface"  
            )       
        }  

    def load_json_dataset(self, config: DatasetConfig) -> DatasetDict:  
        """  
        从本地JSON文件加载数据集  
        
        Args:  
            config: 数据集配置  
            
        Returns:  
            DatasetDict对象  
        """  
        datasets = {}  
        
        # 定义文件路径映射  
        file_paths = { 
            'train': os.path.join(config.local_path, 'train.json'),  
            'validation': os.path.join(config.local_path, 'dev.json'),
            'test': os.path.join(config.local_path, 'test.json')   
               
        }  
        
        for split, file_path in file_paths.items():  
            if not os.path.exists(file_path):  
                print(f"Warning: dataset [{config.name}]'s split [{file_path}] does not exist")  
                continue  
                
            with open(file_path, 'r', encoding='utf-8') as f:  
                data = json.load(f)  
            
            # 针对MultiRC的特殊处理  
            if config.name == "multirc":  
                processed_data = self._process_multirc_json(data, config)  
            # 针对DREAM的特殊处理  
            elif config.name == "dream":  
                processed_data = self._process_dream_json(data,config)
            
            else:  
                processed_data = data  
                
            datasets[split] = Dataset.from_dict(processed_data)  
            
        # merge the splits in the DatasetDict into a single Dataset
        dataset_dict = DatasetDict(datasets)
        
        # splits = dataset_dict.keys()
        # datasets_to_merge = []
        # for split in splits:
        #     ds = dataset_dict[split]
        #     datasets_to_merge.append(ds)
            
        # merged_dataset = concatenate_datasets(datasets_to_merge)
        
        return dataset_dict  

    def _process_multirc_json(self, data: Dict, config:DatasetConfig) -> Dict[str, List]:  
        """  
        处理MultiRC数据集的JSON数据  
        
        MultiRC数据集格式：  
        {  
            "data": [  
                {  
                    "paragraph": {  
                        "text": "...",  # HTML格式的段落文本  
                        "questions": [  
                            {  
                                "question": "...",  
                                "sentences_used": [1, 2, ...],  # 问题相关的句子索引  
                                "answers": [  
                                    {  
                                        "text": "...",  
                                        "isAnswer": true/false,  
                                        "scores": {}  
                                    },  
                                    ...  
                                ],  
                                "idx": "0",  
                                "multisent": true/false  
                            },  
                            ...  
                        ]  
                    }  
                },  
                ...  
            ]  
        }  
        
        Args:  
            data: 原始JSON数据  
            
        Returns:  
            处理后的字典数据，包含以下字段：  
            - context: 清理后的段落文本  
            - question: 问题  
            - options: 候选答案列表  
            - label: 正确答案的索引  
            - sentences_used: 问题相关的句子索引  
            - multisent: 是否跨多个句子  
        """  
        processed_data = {  
            config.article_key: [],  
            config.question_key: [],  
            config.options_key: [],  
            config.label_key: [],  
            "sentences_used": [],  
            "multisent": []  
        }  
        
        def clean_html_text(text: str) -> str:  
            """清理HTML标签和特殊格式"""  
            # 替换HTML标签和特殊字符  
            text = text.replace("<b>", "").replace("</b>", "")  
            text = text.replace("<br>", "\n")  
            # 移除"Sent X: "格式, e.g. Sent 1:  Sent 2:
            text = re.sub(r'Sent \d+: ', '', text)  
            return text.strip()  
        
        # 遍历每个段落  
        for item in data["data"]:  
            paragraph_text = clean_html_text(item["paragraph"]["text"])  
            
            # 处理段落中的每个问题  
            for question_data in item["paragraph"]["questions"]:  
                question = question_data["question"]  
                answers = question_data["answers"]  
                sentences_used = question_data["sentences_used"]  
                multisent = question_data["multisent"]  
                
                # 收集所有答案文本  
                answer_texts = [ans["text"] for ans in answers]  
                
                # 找出正确答案  
                correct_answers = [ans["text"] for ans in answers if ans["isAnswer"]]  
                
                # 如果没有正确答案，跳过这个问题  
                if not correct_answers:  
                    continue  
                    
                # 选择第一个正确答案作为标准答案  
                correct_answer = correct_answers[0]  
                
                # 构建选项列表（确保包含至少一个正确答案）  
                options = [correct_answer]  
                
                # 添加错误答案  
                wrong_answers = [ans["text"] for ans in answers if not ans["isAnswer"]]  
                options.extend(wrong_answers)  
                
                # 如果选项不足4个，用特殊标记填充  
                while len(options) < 4:  
                    options.append("No answer")  
                
                # 如果选项超过4个，只保留4个（确保包含正确答案）  
                if len(options) > 4:  
                    # 保留正确答案和前3个错误答案  
                    wrong_options = [opt for opt in options if opt != correct_answer][:3]  
                    options = [correct_answer] + wrong_options  
                
                # 随机打乱选项顺序  
                original_options = options.copy()  
                random.shuffle(options)  
                
                # 找出正确答案的新索引  
                label = options.index(correct_answer)  
                
                # 添加到处理后的数据中  
                processed_data[config.article_key].append(paragraph_text)  
                processed_data[config.question_key].append(question)  
                processed_data[config.options_key].append(options)  
                processed_data[config.label_key].append(label)  
                processed_data["sentences_used"].append(sentences_used)  
                processed_data["multisent"].append(multisent)  
        
        return processed_data  
  

    def _process_dream_json(self, data:Dict, config:DatasetConfig)->Dict[str, List]: 
        """  
        处理DREAM数据集的JSON数据  
        
        DREAM数据集格式：  
        [  
            [   # dialogue item
                [  
                    "M: ...",  # 对话内容  
                    "W: ..."  
                ],  
                [  
                    {  
                        "question": "...",  
                        "choice": ["choice1", "choice2", "choice3"],  
                        "answer": "correct_answer"  
                    }  
                ],  
                "dialogue_id"  # 例如 "5-510"  
            ],  
            ...  
        ]  
        
        Args:  
            data: 原始JSON数据  
            
        Returns:  
            处理后的字典数据，包含以下字段：  
            - dialogue: 对话文本（合并后）  
            - question: 问题  
            - options: 选项列表  
            - label: 正确答案的索引  
            - dialogue_id: 对话ID（可选，用于追踪）  
        """  
        processed_data = {  
            config.article_key: [],  
            config.question_key: [],  
            config.options_key: [],  
            config.label_key: [],  
            "dialogue_id": []  
        }  
        
        # 遍历每个样本  
        for dialogue_item in data:  
            # 解析数据  
            dialogue_texts = dialogue_item[0]  # 对话内容列表  
            questions = dialogue_item[1]       # 问题列表  
            dialogue_id = dialogue_item[2]     # 对话ID  
            
            # 将对话列表合并成单个文本，每个话语用换行符分隔  
            dialogue_text = "\n".join(dialogue_texts)  
            
            # 处理该对话场景下的所有问题  
            for qa in questions:  
                question = qa["question"]  
                choices = qa["choice"]  
                answer = qa["answer"]  
            
                # 确保选项列表长度为4（DREAM默认是3个选项）  
                options = choices.copy()  
                while len(options) < 3:  
                    options.append("N/A")  # 填充到3个选项  
                    
                # 获取正确答案的索引（在DREAM中答案是选项的完整文本）  
                try:  
                    if isinstance(answer, str) and answer in options and answer.strip()!="":
                        answer_index = options.index(answer)
                        label = chr(65+answer_index) 
                    else:
                        answer_index = random.randint(0,2)
                        label = chr(65+answer_index) 
                        
                except ValueError:  
                    print(f"Warning: Answer '{answer}' not found in choices for dialogue {dialogue_id}")  
                    continue  
                
                labeled_options = [chr(65+i)+". "+option for i, option in enumerate(options)]
                    
                # 添加到处理后的数据中  
                processed_data[config.article_key].append(dialogue_text)  
                processed_data[config.question_key].append(question)  
                processed_data[config.options_key].append(labeled_options)  
                processed_data[config.label_key].append(label)  
                processed_data["dialogue_id"].append(dialogue_id)  
        
        return processed_data   
    
    
    def _process_race(self, data:Dataset, config:DatasetConfig)->Dataset:
        
        
        def process_examples(examples:Dict[str, List]):
            batch_size = len(examples["question"])
            # 初始化结果字典
            result = {
                config.article_key: [],
                config.question_key: [],
                config.options_key: [],
                config.label_key: []
            }
            for i in range(batch_size):
                # 收集所有选项
                options = examples["options"][i]
                labeled_options = [chr(i+65)+". "+option for i, option in enumerate(options)]
                
                correct_answer = examples['answer'][i]
                
                # 添加到结果中  
                result[config.article_key].append(examples["article"][i].strip())  
                result[config.question_key].append(examples["question"][i].strip())  
                result[config.options_key].append(labeled_options)  
                result[config.label_key].append(correct_answer)  
            
            return result
        
        processed_dataset = data.map(  
            process_examples,
            batch_size=Config['batch_size'],
            batched= True,
            num_proc=NUM_CPU_PROCESSES, 
            remove_columns=["example_id"],
            desc="Processing Race dataset", 
        )  
        
        return processed_dataset
        
    
    def _process_sciq(self, data:Dataset, config:DatasetConfig)->Dataset:
        """  
        处理SciQ数据集，将原始格式转换为统一的格式  
        
        原始格式：  
        {  
            "correct_answer": str,  
            "distractor1": str,  
            "distractor2": str,  
            "distractor3": str,  
            "question": str,  
            "support": str  
        }  
        
        目标格式：  
        {  
            "support": str,        # 文章内容  
            "question": str,       # 问题  
            "options": List[str],  # 选项列表（4个选项）  
            "answer": int         # 正确答案的索引  
        }  
        
        Args:  
            data: 原始Dataset对象  
            
        Returns:  
            处理后的Dataset对象  
        """  
        def process_examples(examples:Dict[str, List]):
            batch_size = len(examples["question"]) 
            # 初始化结果字典  
            result = {  
                config.article_key: [],  
                config.question_key: [],  
                config.options_key: [],  
                config.label_key: []  
            }     
            for i in range(batch_size):
                # 收集所有选项  
                options = [  
                    examples["correct_answer"][i].strip(),  
                    examples["distractor1"][i].strip(),  
                    examples["distractor2"][i].strip(),  
                    examples["distractor3"][i].strip(),
                ]    
                

                # 记录正确答案的原始位置（0）  
                correct_answer = examples["correct_answer"][i].strip()
                
                correct_answer_index = options.index(correct_answer)
                
                original_indices = list(range(len(options))) 
                
                combined = list(zip(options, original_indices)) 
                
                # 随机打乱选项顺序  
                random.shuffle(combined)  
                
                shuffled_options, shuffled_indices = zip(*combined) 
                # [1, 3, 2, 0]
                #  0  1  2   3  ->  A  B  C  D
                
                # 添加选项标签 (A, B, C, D)  
                labeled_options = [  
                    f"{chr(j+65)}. {option}" for j, option in enumerate(shuffled_options)  
                ] 
                
                final_answer_index = shuffled_indices.index(correct_answer_index)
                final_answer = chr(final_answer_index+65)
                
                # 添加到结果中  
                result[config.article_key].append(examples["support"][i].strip())  
                result[config.question_key].append(examples["question"][i].strip())  
                result[config.options_key].append(labeled_options)  
                result[config.label_key].append(final_answer)  
                
                
            return result
        
        # 使用map函数处理整个数据集  
        processed_dataset = data.map(  
            process_examples,
            batch_size=Config['batch_size'],
            batched= True,
            num_proc=NUM_CPU_PROCESSES, 
            remove_columns=["correct_answer", "distractor1", "distractor2", "distractor3"],
            desc="Processing SciQ dataset", 
        )  
        
        return processed_dataset  
    
    
    def _process_commonsense_qa(self, data:Dataset, config:DatasetConfig)->Dataset:
        """  
        处理CommonsenseQA数据集，将原始格式转换为统一的格式  
        
        原始格式：  
        {  
            'id': str,  
            'question': str,  [  
                                    'The sanctions against the school were...',  
                                    'What do people typically do when...'  
                                ],  
            
            
            'question_concept': str,  [  
                                            'punishing',  
                                            'walking'  
                                        ],  
            'choices': { 
                {
                
                    'label': List[str],  # ['A', 'B', 'C', 'D', 'E'],
                    'text': List[str]    # 选项文本列表  ['text', 'text', 'text', 'text', 'text'],
                },
                {
                    'label': List[str], 
                    'text': List[str]   
                }
            },  
            'answerKey': str        # 正确答案的标签 ['A', 'C']
        }  
        
        目标格式：  
        {  
            'question_concept': str,  # 文章/概念  
            'question': str,          # 问题  
            'options': List[str],     # 带标签的选项列表 ["A. text", "B. text", ...]  
            'answer': str             # 正确答案标签 ('A'/'B'/...)  
        }  
        
        Args:  
            data: 原始Dataset对象  
            
        Returns:  
            处理后的Dataset对象  
        """  
        def process_examples(examples):  
            
            
            # print("examples = \n", examples)
            batch_size = len(examples["question"])  
            
            # 初始化结果字典  
            result = {  
                config.article_key: [],  
                config.question_key: [],  
                config.options_key: [],  
                config.label_key: []  
            }  
            
            # 处理批次中的每个样本  
            for i in range(batch_size):  
                # 获取选项标签和文本  
                labels = examples["choices"][i]["label"]
                texts = examples["choices"][i]["text"]
                answer = examples['answerKey'][i]  # 可能为空
                
                if isinstance(answer, str) and len(answer.strip()) > 0:  
                    pass 
                else:   
                    choices_num = len(labels)
                    random_num = random.randint(0, choices_num-1)
                    answer = chr(65+random_num)  # 或其他适当的默认值 
                
                # 组合标签和文本  
                labeled_options = [f"{label}. {text}" for label, text in zip(labels, texts)]  
                
                # 添加到结果中  
                result[config.article_key].append(examples["question_concept"][i])  
                result[config.question_key].append(examples["question"][i])  
                result[config.options_key].append(labeled_options)  
                result[config.label_key].append(answer)  
            
            return result  
        
        # 使用map函数处理整个数据集  
        processed_dataset = data.map(  
            process_examples,  
            batched=True,  
            batch_size=32,  
            num_proc=NUM_CPU_PROCESSES,  
            desc="Processing CommonsenseQA dataset", 
            remove_columns=["id", "choices", "answerKey"], 
        )  
        
        return processed_dataset
    
    def load_mcq_dataset(self, dataset_name: str, split = None) -> Tuple[Union[DatasetDict, Dataset], List[str]]:  
        """  
        load a complete dataset [train, valid] 
        
        Args:  
            dataset_name:  
            
        Returns:  
            a dataset that no matter what type they used to be [json, Dataset]
            but now are "huggingface Dataset"
        """  
        config = self.dataset_configs[dataset_name.lower()]  
        
        # 根据文件格式选择加载方式  
        if config.file_format == "json":  
            dataset = self.load_json_dataset(config)  
            
            if split == "train":
                dataset = dataset['train']
            elif split == 'validation':
                dataset = dataset['validation'] # dev
            elif split == 'test':
                try:
                    dataset = dataset['test']
                except:
                    print("the split {} is not found, use validation instead".format(split))
                    dataset = dataset['validation'] # dev
            elif split == None:
                pass
            else:
                raise ValueError(f"Unsupported dataset split name: {split}, Please select from [train, validation, test]")
                
                
        else:  # huggingface格式  
            dataset = load_dataset_from_huggingface(config.local_path, config.subset, split=split)
            # print("dataset['train][0]:\n", dataset['train'][0])
            if dataset_name.lower() == 'sciq':
                dataset = self._process_sciq(dataset, config)
            elif dataset_name.lower() == 'commonsense_qa':
                dataset = self._process_commonsense_qa(dataset, config)
            elif dataset_name.lower() == 'race':
                print("preprocess race ~~~~~")
                dataset = self._process_race(dataset, config)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}, Please select from [race, sciq, commonsense_qa]")

        four_column_names = [config.article_key, config.question_key, config.options_key, config.label_key]
        return dataset, four_column_names

    def _preprocess_multirc(self, dataset: DatasetDict, config: DatasetConfig) -> DatasetDict:  
        """  
        处理MultiRC数据集的特定预处理方法  
        
        MultiRC数据集特点：  
        - 每个段落(paragraph)可能包含多个问题  
        - 每个问题可能有多个候选答案  
        - 每个答案都有一个二元标签(0或1)表示是否正确  
        
        Args:  
            dataset: 原始数据集  
            config: 数据集配置  
            
        Returns:  
            处理后的数据集  
        """  
        def process_multirc_split(examples):  
            # 创建新的特征字典  
            new_features = {  
                "context": [],  # 段落文本  
                "question": [], # 问题  
                "options": [],  # 候选答案列表  
                "label": []     # 正确答案的索引  
            }  
            
            # 对每个样本进行处理  
            for paragraph, question, answers, labels in zip(  
                examples[config.article_key],  
                examples[config.question_key],  
                examples[config.options_key],  
                examples[config.label_key]  
            ):  
                # MultiRC的答案可能是一个字符串或一个列表  
                if isinstance(answers, str):  
                    answers = [answers]  
                if isinstance(labels, (int, bool)):  
                    labels = [labels]  
                    
                # 确保答案和标签长度匹配  
                assert len(answers) == len(labels), "Answers and labels must have the same length"  
                
                # 如果有多个正确答案，我们只取第一个作为正确答案  
                # 这是为了适应多选一的格式  
                if sum(labels) > 0:  
                    correct_idx = labels.index(1)  
                else:  
                    correct_idx = 0  # 如果没有正确答案，默认选第一个  
                    
                # 构建选项列表（确保至少有4个选项）  
                options = answers.copy()  
                while len(options) < 4:  
                    options.append("No answer")  # 填充选项  
                    
                new_features["context"].append(paragraph)  
                new_features["question"].append(question)  
                new_features["options"].append(options[:4])  # 只取前4个选项  
                new_features["label"].append(correct_idx)  
                
            return new_features  
        
        # 对训练集和验证集分别进行处理  
        processed_dataset = DatasetDict({  
            split: dataset[split].map(  
                process_multirc_split,  
                batched=True,  
                remove_columns=dataset[split].column_names  
            )  
            for split in dataset.keys()  
        })  
        
        return processed_dataset  

        
    def get_all_datasets(self, split=None) -> Dict[str, DatasetDict]:  
        """处理所有数据集"""  
        all_datasets = {}  
        for dataset_name in self.dataset_configs.keys():  
            all_datasets[dataset_name], _ = self.load_mcq_dataset(dataset_name, split=split)  
        return all_datasets  




############# define 2 factories ##################################



def preprocess_func_peft(dataset_name, examples, wrapper: McqDatasetWrapper, first_four_columns: List[str], max_length)->Dict[str,Union[List,List[List]]]:
    '''
    预处理函数：将article、question和options字段合并成新的question字段  
           [use tokenization]
           [only used for PEFT tasks]
    Args:  
        examples:Dict[ str, List ]: 数据集中的一个批次样本  
        dataset_name: 数据集名称，用于选择预处理函数  
    Returns:  
        examples after preprocessing       
    '''
    
    if dataset_name == 'race':
        # model_inputs = preprocess_function_race(examples, text_column = "article", label_column  ="answer", 
        #                                  dataset_name = 'race', max_length = max_length)
        
        model_inputs = preprocess_function_race(examples, first_four_columns = first_four_columns, 
                                    dataset_name = 'race', max_length = max_length, tokenizer = wrapper.tokenizer)
    elif dataset_name == 'multirc':
        model_inputs = preprocess_function_multirc(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'multirc', max_length = max_length)
    elif dataset_name == 'arc':
        model_inputs = preprocess_function_arc(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'arc', max_length = max_length)
    elif dataset_name == 'dream':
        model_inputs = preprocess_function_dream(examples, first_four_columns = first_four_columns, 
                                         dataset_name = 'dream', max_length = max_length, tokenizer = wrapper.tokenizer)
    elif dataset_name == 'sciq':
        model_inputs = preprocess_function_sciq(examples, first_four_columns = first_four_columns, 
                                         dataset_name = 'sciq', max_length = max_length, tokenizer = wrapper.tokenizer)
        
    elif dataset_name == 'commonsense_qa':
        model_inputs = preprocess_function_commonsense_qa(examples, first_four_columns = first_four_columns, 
                                         dataset_name = 'commonsense_qa', max_length = max_length, tokenizer = wrapper.tokenizer)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}, please select from [race, multirc, arc, dream, sciq, commonsense_qa]")

    return model_inputs



def preprocess_dataset_peft(dataset_name, max_length=512)->Dataset:
    """  
    处理整个数据集  [dataset必须同时包含train和valid] [针对PEFT任务]
                    # train and valid will be put to dataloader for training and evaluation
    Args:  
        dataset: 原始的训练集 (split = None)  
        dataset_name: 数据集名称，用于选择预处理函数
    Returns:  
        
        preprocessed_dataset: 处理后的数据集，包含train和valid两个部分 
    """ 
    wrapper = McqDatasetWrapper()
    dataset, first_four_columns = wrapper.load_mcq_dataset(dataset_name)
    processed_dataset:DatasetDict = dataset.map(
        function= lambda examples: preprocess_func_peft(dataset_name, examples, wrapper, first_four_columns, max_length),
        batched=True,
        batch_size=Config['batch_size'],
        num_proc=NUM_CPU_PROCESSES,
        remove_columns= dataset['train'].column_names,           # dataset.column_names,
        load_from_cache_file=False,
        desc=f"Running tokenizer on dataset {dataset_name}",
    )
    print(f"\nProcessed dataset type: {type(processed_dataset)}")
    # name = processed_dataset.info.dataset_name if hasattr(processed_dataset.info, 'dataset_name') else None 
    # print(f"Processed dataset column names: {column_names}")  
    print(f"Processed dataset size: {len(processed_dataset)}")  
    if hasattr(processed_dataset, 'column_names'):  
        print(f"Processed dataset columns: {processed_dataset.column_names}")  

    return processed_dataset



if __name__ == "__main__":  
    # create_csv("mcq_dataset.csv", num_examples=1000)  
    
    # dataset = load_dataset_from_csv("mcq_dataset.csv")  
    # dataset = prepare_dataset(dataset)  
    # # 划分训练集和验证集  
    # train_testvalid = dataset.train_test_split(test_size=0.2)  
    # train_dataset = train_testvalid['train']  
    # valid_dataset = train_testvalid['test']  
    
    
    # print("=============================")
    # dataset_path = Config["datasets"]["race"]
    # ds=load_dataset_from_huggingface(dataset_path, "high")
    
    # print("ds['train'][0]  = \n", ds['train'][0])
    # print("answer = ", ds["train"].features)
    
    # ds_builder = load_dataset_builder(dataset_path, "high")
    
    # print("====================")
    # print(ds_builder.info.dataset_name) 
    
    
    # print("====================") 
 
    # preprocess_race(ds)
    # preprocess_pipeline_pt(ds)
    
    
    # dataset_name = "multirc"
    # train_data_path = Config['datasets']['multirc']['train']
    # validation_data_path = Config['datasets']['multirc']['validation']
    # ds = load_dataset_from_json(train_data_path=train_data_path, validation_data_path=validation_data_path, split="train")
    
    # print(ds[0])
    # print("=====================================")
    # print('len(ds[0]) = ',len(ds[0]))
    
    
    # ds = choose_dataset("multirc")
    
    
    # mcqobj = McqDatasetWrapper()
    
    # dream, _ = mcqobj.load_mcq_dataset("dream")
    
    # sciq, _ = mcqobj.load_mcq_dataset("sciq")
    
    # race, _ = mcqobj.load_mcq_dataset("race")
    
    
    # commonsense_qa, _ = mcqobj.load_mcq_dataset("commonsense_qa")
    
    
    # print(" ================= dream =====================")
    
    # print(dream['train'][0])
    
    # print(" ")
    # print("======================== sciq ========================")
    # print(sciq['train'][0])  
    
    
    # print(" ")
    # print("======================== commonsense_qa ========================")
    # print(commonsense_qa['train'][0])
    # print(" ===================================== ")

    
    
    # print(" ")
    # print(" ================= race ==================== ")
    # print(race['train'][0])  
    
    # print(" ===================================== ")
    # print(race['validation'][0])
    
    
    # print("======================================")
    # print(race['test'][0])
    
    
    
    # ds = preprocess_dataset_peft("sciq")
    # ds = preprocess_dataset_peft("commonsense_qa")
    ds = preprocess_dataset_peft("dream")
    
    
    
    print(ds['train'][0])