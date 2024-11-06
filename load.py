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
    AutoModel,
    PreTrainedTokenizer,  
    BatchEncoding,  
)
import random
import torch
import json
import os

from config import Config
from config import NUM_PROCESSES, NUM_CPU_PROCESSES

from typing import List, Dict, Union, Any, Optional



# 初始化分词器  
tokenizer = AutoTokenizer.from_pretrained(Config["models"]["bert-base-uncased"]["model_path"])  





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

def load_dataset_from_csv(file_path):  
    """  
    加载CSV格式的数据集，包含 'context', 'question', 'choices', 'label' 列  
    """  
    df = pd.read_csv(file_path)  
    dataset = Dataset.from_pandas(df)  
    
    print(dataset[0])
    return dataset  

def load_dataset_from_json(train_data_path = None, validation_data_path=None, data_files:Union[str, Dict[str,str]]=None, split = None):
    
    if data_files is not None:
        
        if isinstance(data_files, dict):
            print("data_files is a dict")
        else:
            print("data_files is a str")
        
        ds = load_dataset('json', data_files=data_files, split=split)  

        print(ds[0])
    else:
        if train_data_path and validation_data_path:
            ds = load_dataset('json', data_files={'train':train_data_path, 'validation':validation_data_path}, split= split)
        else:
            if train_data_path:
                if split!='train':
                    raise ValueError(f"train_data_path and split={split} are not compatible !!!")
                ds = load_dataset('json', data_files=train_data_path)
            elif validation_data_path:
                if split!='validation':
                    raise ValueError(f"dev_data_path and split={split} are not compatible !!!")
                ds = load_dataset('json', data_files=validation_data_path)
            else:
                raise ValueError("train_data_path or dev_data_path or data_files must be provided !!!")
        
    return ds
def preprocess_function(examples):  
    """  
    将输入文本进行分词编码  
    """  
    
    # print("example['context'] = ", examples['context'])
    # print("example['question'] = ", examples['question'])
    # print("example['choices'] = ", examples['choices'])
    # print("example['label'] = ", examples['label'])
    
    prompts = [examples['context'][i]+ examples['question'][i]+ examples['choices'][i] for i in range(len(examples['choices']))]
    model_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)  
    model_inputs["labels"] = examples["label"]  
    return model_inputs


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
    

def preprocess_function_race(examples, text_column = "article", label_column  ="answer", 
                             dataset_name = 'race', max_length = 512, tokenizer = None)->Dict[str,Union[List,List[List]]]:
    
    
    
    ''' 
      iteratively modify every batch of data
      
      Args:
          text_column: the column name of the text
          
          ds_builder: use it to get the name of dataset
          
          classes: the set of labels (str) e.g. ['A', 'B', 'C', 'D']    ['good', 'bad']
        
    return: 
  
          {  
                'input_ids': [[101, ..., 102], [101, ..., 102], ...],          # 每个子列表对应一个样本的 token IDs  
                'attention_mask': [[1, ..., 1], [1, ..., 0], ...],             # 1 表示实际 token，0 表示填充  
                'labels': [0, 2, 3, ...]                                        # 标签的整数索引表示  
            }  
          
    '''
    assert tokenizer is not None, "tokenizer in \"preprocess_function_race\" is None, please assign one"
    
    batch_size = len(examples[text_column])
    
    
    # 构建输入文本  
    inputs = [  
        f"Article: {examples['article'][i]}\n\nQuestion: {examples['question'][i]}\n\nOptions: {examples['options'][i]}\n\nAnswer:"  
        for i in range(len(examples[text_column]))  
    ]  
    
    model_inputs = tokenizer(  
        inputs,  
        padding = "max_length",
        truncation=True,  
        max_length=max_length,  
        add_special_tokens=True  # 确保添加特殊标记  [CLS] [SEP]
    )  


    # 处理标签：将答案从字母转换为整数索引  
    labels = [ord(answer) - ord('A') for answer in examples['answer']]  
    
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




def preprocess_function_record(examples, text_column = "article", label_column  ="answer", 
                               dataset_name = 'record', max_length = 512)->Dict[str,Union[List,List[List]]]:
    pass



def preprocess_func_peft(dataset_name, examples, max_length)->Dict[str,Union[List,List[List]]]:
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
        model_inputs = preprocess_function_race(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'race', max_length = max_length)
    elif dataset_name == 'multirc':
        model_inputs = preprocess_function_multirc(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'multirc', max_length = max_length)
    elif dataset_name == 'arc':
        model_inputs = preprocess_function_arc(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'arc', max_length = max_length)
    elif dataset_name == 'record':
        model_inputs = preprocess_function_record(examples, text_column = "article", label_column  ="answer", 
                                         dataset_name = 'record', max_length = max_length)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}, please select from [race, multirc, arc, record]")

    return model_inputs
def preprocess_dataset_peft(dataset_name, dataset:Dataset, max_length=512)->Dataset:
    """  
    处理整个数据集  [dataset必须同时包含train和valid] [针对PEFT任务]
                    # train and valid will be put to dataloader for training and evaluation
    Args:  
        dataset: 原始的训练集 (split = None)  
        dataset_name: 数据集名称，用于选择预处理函数
    Returns:  
        
        preprocessed_dataset: 处理后的数据集，包含train和valid两个部分 
    """ 
    processed_dataset = dataset.map(
        function= lambda examples: preprocess_func_peft(dataset_name, examples, max_length),
        batched=True,
        num_proc=NUM_CPU_PROCESSES,
        remove_columns= ds['train'].column_names,           # dataset.column_names,
        load_from_cache_file=False,
        desc=f"Running tokenizer on dataset {dataset_name}",
    )

    print(f"\nProcessed dataset type: {type(processed_dataset)}")
    name = processed_dataset.info.dataset_name if hasattr(processed_dataset.info, 'dataset_name') else None 
    print(f"Processed dataset name: {name}")  
    print(f"Processed dataset size: {len(processed_dataset)}")  
    if hasattr(processed_dataset, 'column_names'):  
        print(f"Processed dataset columns: {processed_dataset.column_names}")  

    return processed_dataset
    



def choose_dataset(dataset_name:str, split = None)->Dataset:
    '''
     return a huggingface dataset 

     Args:
        dataset_name: str, dataset name, choose from [race, arc, multirc, record]
        split: str, split of dataset, choose from [train, valid, test]
    Returns:
        ds: Dataset, a huggingface dataset
    '''
    ds = None
    if dataset_name == "race":
        dataset_path = Config["datasets"][dataset_name]
        ds = load_dataset_from_huggingface(dataset_path, "high", split=split)
    elif dataset_name == "arc":
        dataset_path = Config["datasets"][dataset_name]
        ds = load_dataset_from_huggingface(dataset_path, "multiple_choice", split=split)
    elif dataset_name == "multirc":
        cache_dir = Config["datasets"]["super_glue"]
        ds = load_dataset_from_huggingface("super_glue", "multirc", split=split, cache_dir=cache_dir)
    elif dataset_name == "record":
        cache_dir = Config["datasets"]["super_glue"]
        ds = load_dataset_from_huggingface("super_glue", "record", split=split, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}, please select from [race, arc, multirc, record]")
    
  
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





def preprocess_func_autocot(dataset_name, examples):
    '''
    预处理函数：将article、question和options字段合并成新的question字段  

            no tokenization !
    Args:  
        examples:Dict[ str, List ]: 数据集中的一个批次样本  
        dataset_name: 数据集名称，用于选择预处理函数  
    Returns:  
        处理后的样本     
    '''
    
    if dataset_name == "race":
        articles = [str(art) for art in examples['article']]  
        questions = [str(q) for q in examples['question']]  
        options = [str(opt) for opt in examples['options']] 
        
        new_questions: List[str] = [  
            f"Artical:{art}\nQuestion:{q}\nOptions:{opt}"   
            for art, q, opt in zip(articles, questions, options)  
        ]  
        
        examples['question'] = new_questions 
        
        examples.pop('article')  
        examples.pop('options')  
        return examples
    
    elif dataset_name == 'record':
        pass
    elif dataset_name == 'arc':
        pass
    elif dataset_name == 'multirc':
        pass
    else:
        raise ValueError("Invalid dataset name ... Please select from [race, record, multirc, arc]")



def preprocess_dataset_autocot(dataset_name, dataset:Dataset):
    """  
    处理整个数据集  [必须是训练集]
    
    Args:  
        dataset: 原始的训练集 (split = train)  
        dataset_name: 数据集名称，用于选择预处理函数
    Returns:  
        处理后的数据集  
    """ 
    
    # 先预处理第一个样本试试看 
    print("\nTesting preprocess_func_autocot with first example...")  
    try:  
        first_example = dataset[0]  
        print(f"First example before processing: {first_example}")  
        processed_example = preprocess_func_autocot(dataset_name, {'examples': [first_example]})  
        print(f"First example after processing: {processed_example}")  
    except Exception as e:  
        print(f"Error in test first example: {str(e)}")  
        import traceback  
        print(f"Traceback:\n{traceback.format_exc()}")  
    
    print("\nFirst example testing succeed, now, Starting dataset mapping...")  
    processed_dataset = dataset.map(
        function= lambda examples: preprocess_func_autocot(dataset_name, examples),
        batched=True,
        num_proc=NUM_CPU_PROCESSES,
        remove_columns= ["example_id", "article", "options"],           # dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    print(f"\nProcessed dataset type: {type(processed_dataset)}")
    name = processed_dataset.info.dataset_name if hasattr(processed_dataset.info, 'dataset_name') else None 
    print(f"Processed dataset name: {name}")  
    print(f"Processed dataset size: {len(processed_dataset)}")  
    if hasattr(processed_dataset, 'column_names'):  
        print(f"Processed dataset columns: {processed_dataset.column_names}")  

    return processed_dataset









########## Load and Reformat all datasets for PEFT tasks ###############



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


class MultipleChoicePreprocessor:  
    '''
     This class provides methods for preprocessing multiple choice datasets.
     
     with different format {"huggingface", "json"}
    '''
    def __init__(  
        self,  
        model_name_or_path: str = "bert-base-uncased",  
        max_seq_length: int = 512,  
        label_map: Dict[str, int] = None  
    ):  
        """  
        初始化预处理器  
        
        Args:  
            model_name_or_path: BERT模型名称  
            max_seq_length: 最大序列长度  
            label_map: 标签映射字典，例如 {"A": 0, "B": 1, "C": 2, "D": 3}  
        """  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  
        self.max_seq_length = max_seq_length  
        self.label_map = label_map or {"A": 0, "B": 1, "C": 2, "D": 3}  
        
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
            "record": DatasetConfig(  
                name="record",  
                article_key="passage",  
                question_key="query",  
                options_key="entities",  
                label_key="answers",  
                local_path=Config['datasets']['record']['all'],  
                file_format="json"  
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
            'test': os.path.join(config.local_path, 'dev.json')   
        }  
        
        for split, file_path in file_paths.items():  
            if not os.path.exists(file_path):  
                print(f"Warning: dataset [{config.name}]'s split [{file_path}] does not exist")  
                continue  
                
            with open(file_path, 'r', encoding='utf-8') as f:  
                data = json.load(f)  
            
            # 针对MultiRC的特殊处理  
            if config.name == "multirc":  
                processed_data = self._process_multirc_json(data)  
            # 针对ReCoRD的特殊处理  
            elif config.name == "record":  
                processed_data = self._process_record_json(data)  
            else:  
                processed_data = data  
                
            datasets[split] = Dataset.from_dict(processed_data)  
        
        return DatasetDict(datasets)  

    def _process_multirc_json(self, data: Dict) -> Dict[str, List]:  
        """  
        处理MultiRC的JSON数据  
        
        Args:  
            data: 原始JSON数据  
            
        Returns:  
            处理后的字典数据  
        """  
        processed_data = {  
            "paragraph": [],  
            "question": [],  
            "answer": [],  
            "label": []  
        }  
        
        for article in data['data']:  
            for paragraph in article['paragraphs']:  
                for question in paragraph['questions']:  
                    for answer in question['answers']:  
                        processed_data['paragraph'].append(paragraph['text'])  
                        processed_data['question'].append(question['question'])  
                        processed_data['answer'].append(answer['text'])  
                        processed_data['label'].append(int(answer['label']))  
                        
        return processed_data  

    def _process_record_json(self, data: Dict) -> Dict[str, List]:  
        """  
        处理ReCoRD的JSON数据  
        
        Args:  
            data: 原始JSON数据  
            
        Returns:  
            处理后的字典数据  
        """  
        processed_data = {  
            "passage": [],  
            "query": [],  
            "entities": [],  
            "answers": []  
        }  
        
        for item in data['data']:  
            passage = item['passage']['text']  
            entities = item['passage']['entities']  
            
            for qa in item['qas']:  
                processed_data['passage'].append(passage)  
                processed_data['query'].append(qa['query'])  
                processed_data['entities'].append(entities)  
                processed_data['answers'].append(qa['answers'])  
                
        return processed_data  

    def load_and_preprocess_dataset(self, dataset_name: str) -> DatasetDict:  
        """  
        加载并预处理指定的数据集  
        
        Args:  
            dataset_name: 数据集名称  
            
        Returns:  
            处理后的数据集  
        """  
        config = self.dataset_configs[dataset_name.lower()]  
        
        # 根据文件格式选择加载方式  
        if config.file_format == "json":  
            dataset = self.load_json_dataset(config)  
        else:  # huggingface格式  
            # dataset = load_dataset(  
            #     "json",   
            #     data_files=None,  # 使用默认的数据文件  
            #     data_dir=config.local_path  
            # )  
            
            dataset = load_dataset_from_huggingface(config.local_path, config.subset)
            
        # 数据集特定的预处理  
        if dataset_name.lower() == "race":  
            dataset = self._preprocess_race(dataset, config)  
        elif dataset_name.lower() == "ai2_arc":  
            dataset = self._preprocess_arc(dataset, config)  
        elif dataset_name.lower() == "multirc":  
            dataset = self._preprocess_multirc(dataset, config)  
        elif dataset_name.lower() == "record":  
            dataset = self._preprocess_record(dataset, config)  
            
        # 统一格式化处理  
        dataset = dataset.map(  
            self._convert_to_features,  
            batched=True,  
            remove_columns=dataset["train"].column_names  
        )  
        
        return dataset  
    
    def _preprocess_race(self, dataset: DatasetDict, config: DatasetConfig) -> DatasetDict:  
        """RACE数据集的特定预处理"""  
        def process_race(example):  
            return {  
                "context": example[config.article_key],  
                "question": example[config.question_key],  
                "options": example[config.options_key],  
                "label": self.label_map[example[config.label_key]]  
            }  
        return dataset.map(process_race)  

    def _preprocess_arc(self, dataset: DatasetDict, config: DatasetConfig) -> DatasetDict:  
        """ARC数据集的特定预处理"""  
        def process_arc(example):  
            options = [choice["text"] for choice in example[config.options_key]]  
            return {  
                "context": example[config.article_key],  
                "question": example[config.question_key],  
                "options": options,  
                "label": self.label_map[example[config.label_key]]  
            }  
        return dataset.map(process_arc)  
    
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

    def _preprocess_record(self, dataset: DatasetDict, config: DatasetConfig) -> DatasetDict:  
        """  
        处理ReCoRD数据集的特定预处理方法  
        
        ReCoRD数据集特点：  
        - 每个样本包含一段文本(passage)和多个实体(entities)  
        - 问题(query)中包含@placeholder标记，需要用实体替换  
        - 每个问题可能有多个正确答案  
        
        Args:  
            dataset: 原始数据集  
            config: 数据集配置  
            
        Returns:  
            处理后的数据集  
        """  
        def process_record_split(examples):  
            new_features = {  
                "context": [],   # 段落文本  
                "question": [],  # 处理后的问题  
                "options": [],   # 候选答案列表  
                "label": []      # 正确答案的索引  
            }  
            
            for passage, query, entities, answers in zip(  
                examples[config.article_key],  
                examples[config.question_key],  
                examples[config.options_key],  
                examples[config.label_key]  
            ):  
                # 确保entities是列表类型  
                if isinstance(entities, str):  
                    entities = [entities]  
                
                # 过滤出在文章中出现的实体  
                valid_entities = [  
                    entity for entity in entities  
                    if entity in passage  
                ]  
                
                # 如果没有有效实体，跳过这个样本  
                if not valid_entities:  
                    continue  
                    
                # 从有效实体中选择候选答案  
                # 优先使用正确答案，然后随机选择其他实体作为干扰项  
                candidates = set(answers) & set(valid_entities)  
                distractors = set(valid_entities) - set(answers)  
                
                # 构建选项列表  
                options = list(candidates)[:1]  # 取一个正确答案  
                options.extend(list(distractors)[:3])  # 添加最多3个干扰项  
                
                # 如果选项不足4个，用特殊标记填充  
                while len(options) < 4:  
                    options.append("[NO_ENTITY]")  
                    
                # 打乱选项顺序  
                import random  
                random.shuffle(options)  
                
                # 找出正确答案的索引  
                correct_answer = list(candidates)[0]  
                correct_idx = options.index(correct_answer)  
                
                # 替换问题中的占位符  
                processed_query = query.replace("@placeholder", "___")  
                
                new_features["context"].append(passage)  
                new_features["question"].append(processed_query)  
                new_features["options"].append(options)  
                new_features["label"].append(correct_idx)  
            
            return new_features  
        
        # 对训练集和验证集分别进行处理  
        processed_dataset = DatasetDict({  
            split: dataset[split].map(  
                process_record_split,  
                batched=True,  
                remove_columns=dataset[split].column_names  
            )  
            for split in dataset.keys()  
        })  
        
        return processed_dataset  

    def _convert_to_features(self, examples: Dict[str, List[Any]]) -> BatchEncoding:  
        """  
        将数据集转换为模型输入特征  
        
        Args:  
            examples: 批量样本  
            
        Returns:  
            模型输入特征  
        """  
        first_sentences = [[context] * len(options) for context, options in zip(examples["context"], examples["options"])]  
        second_sentences = []  
        
        for question, options in zip(examples["question"], examples["options"]):  
            second_sentences.append(  
                [f"{question} {opt}" for opt in options]  
            )  
        
        # 展平用于批处理  
        first_sentences = sum(first_sentences, [])  
        second_sentences = sum(second_sentences, [])  
        
        # 标记化  
        tokenized_examples = self.tokenizer(  
            first_sentences,  
            second_sentences,  
            truncation="longest_first",  
            max_length=self.max_seq_length,  
            padding="max_length",  
            return_tensors="pt"  
        )  
        
        # 重塑为 (batch_size, num_choices, seq_length)  
        input_ids = tokenized_examples["input_ids"].view(-1, len(examples["options"][0]), self.max_seq_length)  
        attention_mask = tokenized_examples["attention_mask"].view(-1, len(examples["options"][0]), self.max_seq_length)  
        token_type_ids = tokenized_examples["token_type_ids"].view(-1, len(examples["options"][0]), self.max_seq_length)  
        
        return {  
            "input_ids": input_ids,  
            "attention_mask": attention_mask,  
            "token_type_ids": token_type_ids,  
            "labels": examples["label"]  
        }  

    def process_all_datasets(self) -> Dict[str, DatasetDict]:  
        """处理所有数据集"""  
        processed_datasets = {}  
        for dataset_name in self.dataset_configs.keys():  
            processed_datasets[dataset_name] = self.load_and_preprocess_dataset(dataset_name)  
        return processed_datasets  







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
    
    
    ds = choose_dataset("multirc")