import pandas as pd  
from datasets import Dataset, DatasetBuilder, load_dataset, load_dataset_builder
from transformers import AutoTokenizer, AutoModel
import random
import torch

from config import Config



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
    

def preprocess_function_race(examples, text_column = "article", label_column  ="answer", dataset_name = 'race', max_length = 512, tokenizer = None):
    
    
    
    ''' 
      iteratively modify every batch of data
      
      Args:
          text_column: the column name of the text
          
          ds_builder: use it to get the name of dataset
          
          classes: the set of labels (str) e.g. ['A', 'B', 'C', 'D']    ['good', 'bad']
          
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

def preprocess_race_h(ds:Dataset):
        
    print(ds["train"].features)
    # classes = [k.strip() for k in ds["train"].features["answer"].names]
    
    # print("classes = ", classes)
    
    # race的Answer 就是 A, B, C, D. no need for mapping
    ds = ds.map(
        lambda x: x,
        batched=True,
        num_proc=1,
    )

    
    print(f"dataset \"race-high\" is ready to be used ~~~")
    return ds


def preprocess_race_m(ds:Dataset):
        
    print(ds["train"].features)
    # classes = [k.strip() for k in ds["train"].features["answer"].names]
    
    # print("classes = ", classes)
    
    # race的Answer 就是 A, B, C, D. no need for mapping
    ds = ds.map(
        lambda x: x,
        batched=True,
        num_proc=1,
    )

    
    print(f"dataset \"race-middle\" is ready to be used ~~~")
    return ds






def preprocess_race_c(dataset: Dataset):
    pass

def preprocess_mnli(dataset: Dataset):
    pass


def preprocess_mrpc(dataset: Dataset):
    pass






def prepare_dataset(dataset):  
    """  
    应用预处理函数  
    """  
    dataset = dataset.map(preprocess_function, batched=True)  
    return dataset



def generate_example_data(num_examples=1000):  
    data = {  
        "context": [],  
        "question": [],  
        "choices": [],  
        "label": []  
    }  
    
    for _ in range(num_examples):  
        context = "This is a sample context about a certain topic."  
        question = "What is the main idea of the context?"  
        choices = "A) Idea 1 B) Idea 2 C) Idea 3 D) Idea 4"  
        label = random.choice(["A", "B", "C", "D"])  # 随机选择一个正确答案  
        
        data["context"].append(context)  
        data["question"].append(question)  
        data["choices"].append(choices)  
        data["label"].append(label)  
    
    return data  


def create_csv(file_path, num_examples=1000):  
    data = generate_example_data(num_examples)  
    df = pd.DataFrame(data)  
    df.to_csv(file_path, index=False)  
    print(f"CSV file with {num_examples} examples created at {file_path}") 


if __name__ == "__main__":  
    # create_csv("mcq_dataset.csv", num_examples=1000)  
    
    # dataset = load_dataset_from_csv("mcq_dataset.csv")  
    # dataset = prepare_dataset(dataset)  
    # # 划分训练集和验证集  
    # train_testvalid = dataset.train_test_split(test_size=0.2)  
    # train_dataset = train_testvalid['train']  
    # valid_dataset = train_testvalid['test']  
    
    
    print("=============================")
    dataset_path = Config["datasets"]["race"]
    ds=load_dataset_from_huggingface(dataset_path, "high")
    
    print("ds['train'][0]  = \n", ds['train'][0])
    print("answer = ", ds["train"].features)
    
    ds_builder = load_dataset_builder(dataset_path, "high")
    
    print("====================")
    print(ds_builder.info.dataset_name) 
    
    
    print("====================") 
 
    # preprocess_race(ds)
    # preprocess_pipeline_pt(ds)