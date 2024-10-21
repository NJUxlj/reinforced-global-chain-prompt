import pandas as pd  
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer 
import random

from config import Config



# 初始化分词器  
tokenizer = AutoTokenizer.from_pretrained(Config["models"]["bert"]["model_path"])  



def load_dataset_from_huggingface(file_path):
    return load_dataset(file_path)

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


def preprocess_function_pt(examples, text_column = "text", label_column  ="label"):
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
    
    ds=load_dataset_from_huggingface()