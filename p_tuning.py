
import torch
import torch.nn as nn
from torch.optim import Adam
import csv
import evaluate
import numpy as np
from config import Config

from dataclasses import dataclass
from load import *
from utils import *
from typing import Tuple, List, Tuple, Optional, Union, Dict, Any

from torch.utils.data import DataLoader

from datasets import (
    Dataset,
    load_dataset
)


from transformers import (
    set_seed,
    default_data_collator,
    AutoModel,
    BertModel,
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)

from peft import (
    TaskType,
    PeftType,
    PromptEncoder,
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


from accelerate import (
    Accelerator,
)

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


# we follow the setting of prompt-tuning (Lester et al., 2021)
@dataclass  
class PtuningConfig:  
    """MCQA任务的P-tuning V2配置"""  
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "p-tuning"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 100                        # 前缀长度  
    num_labels: int = 4                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 2
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                     # 最大序列长度  
    learning_rate: float = 0.3                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768    
    encoder_hidden_size:int = 512  # 重参数化器MLP的隐藏层大小# prefix token 的维度 (P_theta')
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减  
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # beta2: 二阶矩估计的指数衰减率（默认0.999）
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = Adam


def train_p_tuning(config:PtuningConfig):
    model_name = config.model_name
    model, tokenizer = prepare_model_tokenizer(config.model_path, AutoModelForSequenceClassification, config.model_path )
    dataset_name = config.dataset_name
    # 初始化参数  
    # num_labels = 4  # ['A', 'B', 'C', 'D']
    num_labels = config.num_labels
    prefix_length = config.prefix_length 
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    lr = config.learning_rate
    max_length = config.max_seq_length
    
    print(f"before inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    max_length = max_length - prefix_length # 实际的输入长度
    print(f"After inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    
    # wrapper = McqDatasetWrapper()
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
    device = Config['device'] 
    
    # Prompt-tuning
    peft_config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type= TaskType.SEQ_CLS, 
        num_virtual_tokens=prefix_length, 
        token_dim=config.prefix_hidden_size, # 与基础模型bert保持一致
        # num_transformer_submodules=1,
        # num_attention_heads=12,
        # num_layers=12,
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_num_layers=2,
        encoder_dropout= 0.1,
        inference_mode=False   # 训练模式
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    model = get_peft_model(model, peft_config)
    
    
    # make sure to frozen the base model parameters
    for param in model.base_model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix tokens is trainable
    for name, param in model.named_parameters():  
        if 'prefix_encoder' in name:  
            param.requires_grad = True
    
    
    model.print_trainable_parameters()
    



    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=config.weight_decay, 
        betas=(config.beta1_decay, config.beta2_decay)
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    accelerator = Accelerator(
        # gradient_accumulation_steps=1,  
        # mixed_precision='fp16', 
    )
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)

    print("type(model) = ", type(model))
    
    if accelerator.is_main_process:
        logging_dir = Config['logging_dir'][model_name]["p-tuning"][dataset_name]
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name="log_eval", logging_dir=logging_dir, log_level="INFO")
    

    
    
    device = Config['device']
    global_step = 0

    # 定义一个列表来存储评估结果  
    evaluation_results = []  

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            labels = batch["labels"]  
            outputs = model(**batch)
            criterion = nn.CrossEntropyLoss()
            
            logits = outputs.logits
            loss = criterion(logits, labels.long())
            total_loss += loss.detach().float()
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if step == len(train_dataloader) - 1:  
                model.eval()  
                all_preds = []  
                all_labels = []  
                with torch.no_grad():  
                    for val_batch in eval_dataloader:  
                        val_input_ids = val_batch['input_ids']
                        val_attention_mask = val_batch['attention_mask']
                        val_labels = val_batch['labels'] 
                        val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)  
                        logits = val_outputs['logits']  
                        preds = torch.argmax(logits, dim=1).cpu().numpy()  
                        labels_cpu = val_labels.cpu().numpy()  
                        all_preds.extend(preds)  
                        all_labels.extend(labels_cpu)  
                # 计算评价指标  
                accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')  
                print(f"Step {global_step}, Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")  

                if accelerator.is_main_process:
                    logger.info({'epoch': epoch, 'loss': loss.item(), 'accuracy':accuracy, "precision": precision, "recall": recall, "f1": f1 })  
                    # print()

                model.train()  
            global_step+=1

        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  
            
    
    # 保存模型
    
    print("model name = ", model_name)
    save_path = Config['save_model_dir'][model_name]['p-tuning'][dataset_name]
    
    # wait every GPU processes to reach here
    torch.distributed.barrier()  
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
        print(f"已创建新的权重存储路径: {save_path}") 
    
    # if accelerator.is_main_process():
    #     torch.save()

    accelerator.save(model.state_dict(), save_path)  
    
    
    
    
    
    
    
    
def evaluate_p_tuning(config: PtuningConfig):
    # 保存评估结果到 CSV 文件  
    csv_file_path = f'csv/{config.model_name}/{config.peft_method}/{config.dataset_name}/evaluation_output.csv'  

    evaluation_results = {}
    
    # 这里进行评估集的迭代推理
    
    evaluation_results.append({  
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    })  
    

    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:  
        fieldnames = ['peft_method', 'dataset','model_name','accuracy', 'precision','recall','f1']  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)  

        writer.writeheader()  
        for result in evaluation_results:  
            writer.writerow(result)  

    print(f"Evaluation results saved to {csv_file_path}") 
    
    
    

if __name__ == '__main__':
    '''
    
    '''
    model_path = Config["models"]["bert-base-uncased"]["model_path"]

    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)

    max_seq_length = get_max_length_from_model(model)
    
    config = PtuningConfig(
        model_name = "bert-base-uncased",
        model_path = model_path,
        max_seq_length=max_seq_length,
        dataset_name= 'race',
        num_epochs=5,
    )
    
    train_p_tuning(config)
    