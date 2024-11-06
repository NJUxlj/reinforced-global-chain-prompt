import pandas as pd  
from datasets import Dataset, DatasetBuilder, load_dataset, load_dataset_builder
from transformers import AutoTokenizer, AutoModel
import random
import torch

from config import Config
from config import NUM_PROCESSES, NUM_CPU_PROCESSES

from typing import List, Dict, Union



# 初始化分词器  
tokenizer = AutoTokenizer.from_pretrained(Config["models"]["bert-base-uncased"]["model_path"])  



def load_dataset_from_huggingface(dataset_path, subset_name = None, split = None):
    '''
    load dataset from huggingface hub
    
        dataset_path: "/your/path/glue"
        
        subset_name: "sst2"
        
        split: "train", "validation"
    '''
    
    if subset_name:
        ds = load_dataset(dataset_path, subset_name, split=split)
    else:
        ds = load_dataset(dataset_path, split=split)

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
    """
    


def preprocess_function_arc(examples, text_column = "article", label_column  ="answer", 
                            dataset_name = 'arc', max_length = 512)->Dict[str,Union[List,List[List]]]:
    pass




def preprocess_function_record(examples, text_column = "article", label_column  ="answer", 
                               dataset_name = 'record', max_length = 512)->Dict[str,Union[List,List[List]]]:
    pass







def preprocess_pipeline_pt(ds: Dataset):
    '''
        将所有数据预处理流程放到一个函数中
    '''
    classes = []
    
    dataset_name = ds.info.dataset_name 
    
    if dataset_name == "race":
        preprocess_race(ds)
        
    elif dataset_name == "race-c":
        classes = ds.info.features['answer'].names
    
        
    elif dataset_name == "mnli":
        classes = ds.info.features['label'].names
        
    elif dataset_name == "mrpc":
        classes = ds.info.features['label'].names
    
    
    


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
    
    
    dataset_name = "multirc"
    train_data_path = Config['datasets']['multirc']['train']
    validation_data_path = Config['datasets']['multirc']['validation']
    ds = load_dataset_from_json(train_data_path=train_data_path, validation_data_path=validation_data_path, split="train")
    
    print(ds[0])
    print("=====================================")
    print('len(ds[0]) = ',len(ds[0]))