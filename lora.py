import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from config import SAVE_DIR


import argparse
import evaluate
import os

from load import *

from utils import *

from datasets import(
    load_dataset,
    Dataset,
    DatasetBuilder
)

from utils import (
    prepare_model_tokenizer
)



from peft import (
    LoraConfig,
    TaskType,
    PeftType,
    PromptEncoderConfig, 
    get_peft_model, 
    get_peft_config,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEmbedding,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    # AutoPeftModelForMultipleChoice,
    
)


from transformers import(
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)

from accelerate import (
    Accelerator
)

import logging  


from sklearn.metrics import precision_recall_fscore_support  
from tqdm import tqdm



@dataclass  
class LoraTrainerConfig:  
    """MCQA任务的P-tuning V2配置"""  
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "lora"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 10                        # prefix-tuning的默认前缀长度  
    num_labels: int = 4                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 5
    num_epochs:int = 2
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                         # 最大序列长度  
    learning_rate: float = 5e-5                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768               # MLP中的P_theta'  即，MLP输入的隐单元维度  huggingface 默认它==encoder_hidden_size
    prefix_projection_hidden_size:int = 4*prefix_hidden_size  # 论文中重参数化用的MLP层的中间维度是hidden_size的4倍 
    encoder_hidden_size:int = prefix_hidden_size  # 编码器的隐藏层大小
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减 
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # 用于AdaFactor optimizer
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = AdamW 







def compute_metrics_lora(eval_preds):  
    """  
    计算模型在训练过程中的评估指标。  

    参数：  
        eval_preds: 模型预测的结果，包含 logits 和真实标签。  

    返回：  
        包含准确率、精确率、召回率和 F1 分数的字典。  
    """  
    import numpy as np  
    import evaluate  

    # 解包预测结果和真实标签  
    logits, labels = eval_preds  

    # 将 logits 转换为预测的类别索引  
    predictions = np.argmax(logits, axis=-1)  

    # 加载评估指标  
    accuracy_metric = evaluate.load("accuracy")  
    # precision_metric = evaluate.load("precision")  
    # recall_metric = evaluate.load("recall")  
    f1_metric = evaluate.load("f1")  

    # 计算准确率  
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)  
    
    
    precision, recall, f1, _ = precision_recall_fscore_support(  
        labels, predictions, average='weighted'  
    ) 
    # 计算精确率  
    # precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')  
    # 计算召回率  
    # recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')  
    
    # 计算 F1 分数  
    # f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')  

    # 返回评估指标的结果  
    return {  
        'accuracy': accuracy['accuracy'],  
        'precision': precision,  
        'recall': recall,  
        'f1': f1  
    }





def train_lora(config:LoraTrainerConfig):
    
    # 初始化参数  
    model_name = config.model_name
    model, tokenizer = prepare_model_tokenizer(config.model_path, AutoModelForSequenceClassification, config.model_path, config.num_labels)
    
    num_labels = config.num_labels
    prefix_length = config.prefix_length
    batch_size = config.batch_size
    lr = config.learning_rate
    num_epochs = config.num_epochs
    
    max_length = config.max_seq_length
    print(f"before inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    max_length = max_length - prefix_length
    print(f"After inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    
    # 加载数据集
    dataset_name = config.dataset_name
    dataset_path = get_dataset_path_by_name(dataset_name)
    
    processed_ds = preprocess_dataset_peft(dataset_name, max_length=max_length)
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]  
    
    train_dataloader = DataLoader(
            train_ds, 
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            pin_memory=True
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            pin_memory=True
        )
    
    # 初始化模型  
    device = Config['device'] 
    
    
    
    lora_config = LoraConfig(
        peft_type=PeftType.LORA,
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"], # 除了LoRA层以外, 还有哪些模块需要进行训练和保存
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    
    # # preprocess dataset
    # dataset_name = "race"
    # dataset_path = Config["datasets"][dataset_name]
    # ds = load_dataset_from_huggingface(dataset_path,"high")
    
    # # coarse-grained preprocessing
    # ds, classes, tokenizer = preprocess_race(ds, tokenizer)
    
    # Config["classes"][dataset_name] = classes
    
    # fine-grained preprocessing
    # processed_ds = ds.map(
    #     lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), # 从load.py导入  max_length = 492, 等下要加20个virtual tokens
    #     batched=True,
    #     num_proc=1,
    #     remove_columns=ds['train'].column_names,
    #     load_from_cache_file=False,
    #     desc="Running tokenizer on dataset",    
    # )   
    
    # train_ds = processed_ds["train"]
    # eval_ds = processed_ds["test"]
    
    
    
    
    args = TrainingArguments(
        # peft_model_id,
        output_dir=Config["output_dir"],  # 用来存储save_strategy保存的模型检查点
        evaluation_strategy="epoch",   #  将评估策略改为每隔一定步数评估一次 
        # eval_steps = 5,               #  每隔 5 个步骤进行一次评估 
        logging_strategy="epoch",     # 日志记录策略设为每隔一定步数记录一次 
        # logging_steps=100,
        remove_unused_columns=False,
        save_strategy="epoch",    #  每隔一定步数保存一次模型
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        label_names=["labels"],
        logging_dir=f"logs/{model_name}/{config.peft_method}/{dataset_name}/",           # 指定日志目录  
        report_to=["none"],                # 不报告到其他平台，如 TensorBoard 等  
        logging_first_step=False,           # 记录第一个步骤的日志  
        log_level='info',                  # 设置日志级别  
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_lora,
    )
    trainer.train()





if __name__ == "__main__":
    '''
    
    '''
    
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    model_name = "bert-base-uncased"
    dataset_name = "race"
    
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)

    max_seq_length = get_max_length_from_model(model)
    
    
    config = LoraTrainerConfig(
        model_name=model_name,
        model_path=model_path,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
    )
    

    
    train_lora(config)