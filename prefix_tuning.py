import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW, Adam
from config import Config
import numpy as np
import evaluate
import os
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
from typing import Tuple, List, Tuple, Optional, Union, Dict, Any



from load import *

from utils import *


from torch.utils.data import DataLoader
from datasets import (
    Dataset, 
    load_dataset
)

from transformers import (
    set_seed,
    default_data_collator,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)
from peft import (
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

from transformers.modeling_outputs import (
    SequenceClassifierOutput, # 是一种官方的输出格式
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput
)


from accelerate import (
    Accelerator
)

from tqdm import tqdm



'''
Prefix tuning prefixes a series of task-specific vectors to the input sequence that can be learned while keeping the pretrained model frozen. 
The prefix parameters are inserted in all of the model layers.

'''

@dataclass  
class PrefixTuningTrainerConfig:  
    """MCQA任务的P-tuning V2配置"""  
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "prefix-tuning"
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
    
    
    
    
    
def verify_peft_config(model, peft_config:PrefixTuningConfig, prefix_trainer_config:PrefixTuningTrainerConfig):
    """验证模型配置是否正确"""  
    base_model_config = model.config  
    print("Base model config:")  
    print(f"Hidden size: {base_model_config.hidden_size}")  
    print(f"Num attention heads: {base_model_config.num_attention_heads}")  
    print(f"Num hidden layers: {base_model_config.num_hidden_layers}")  
    
    print("\nPEFT config:")  
    print(f"Encoder hidden size: {peft_config.encoder_hidden_size}")  
    print(f"Num attention heads: {peft_config.num_attention_heads}")  
    print(f"Num layers: {peft_config.num_layers}")  
    
    # 验证维度匹配  
    assert base_model_config.hidden_size == peft_config.encoder_hidden_size, "Hidden size mismatch"  
    assert base_model_config.num_attention_heads == peft_config.num_attention_heads, "Num attention heads mismatch"  
    assert base_model_config.num_hidden_layers == peft_config.num_layers, "Num layers mismatch"  
    
    # 1. 检查batch_size  
    print("Batch size in training args:", prefix_trainer_config.batch_size)  
    
    # 2. 检查模型参数  
    for name, param in model.named_parameters():  
        if "prefix" in name:  
            print(f"{name}: {param.shape}")  
    
    # 3. 检查注意力头维度  
    head_dim = config.encoder_hidden_size // peft_config.num_attention_heads  
    print(f"Attention head dimension: {head_dim}")  
    
    # 4. 验证总参数量  
    expected_size = (  
        prefix_trainer_config.batch_size *  
        peft_config.num_virtual_tokens *  
        peft_config.num_layers *  peft_config.num_transformer_submodules *
        peft_config.num_attention_heads *  
        (peft_config.encoder_hidden_size // peft_config.num_attention_heads)  
    )  
    
    '''
    
     past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
    
    
    '''
    print(f"Expected tensor size for [Prefix-Tuning]'s prefix tokens: {expected_size}")  
    

def train_prefix_tuning(config:PrefixTuningTrainerConfig=None):
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
    
    
    # Prefix-tuning
    peft_config = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=prefix_length, 
        token_dim = config.prefix_hidden_size,      
        num_transformer_submodules=2,   # 论文中只对transformer block 中的 K, V 使用了prefix，所以是2而不是1  
        num_attention_heads=12,
        num_layers=12,
        encoder_hidden_size=config.encoder_hidden_size,  # bert隐藏层维度
        prefix_projection=config.prefix_projection, # # 论文中使用了MLP进行prefix投影
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    # 使用 get_peft_model 包装模型时，PEFT 库通常会自动冻结基础模型的参数，并将 PEFT 参数设置为可训练。
    model = get_peft_model(model, peft_config)
    
    verify_peft_config(model, peft_config, config)
    
    # make sure to frozen the base model parameters
    for param in model.base_model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix tokens is trainable
    for name, param in model.named_parameters():  
        if 'prefix_encoder' in name:  
            param.requires_grad = True 
    
    
    model.print_trainable_parameters()
    


    # make sure that the fine-tuning will only update virual tokens
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
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
    
    
    if accelerator.is_main_process:
        logging_dir = Config['logging_dir'][model_name][config.peft_method][dataset_name]
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name="log_eval", logging_dir=logging_dir, log_level="INFO")
    
    
    
    
    device = Config['device']
    global_step = 0
    
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
            
            if step == len(train_dataloader)-1:  
                model.eval()  
                all_preds = []  
                all_labels = []  
                with torch.no_grad():  
                    for val_batch in eval_dataloader:  
                        val_input_ids = val_batch['input_ids']
                        val_attention_mask = val_batch['attention_mask']
                        val_labels = val_batch['labels']
                        val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)  
                        logits = val_outputs['logits']  # batch_size x num_labels
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
            


    # 保存权重
    print("model name = ", model_name)
    save_path = Config['save_model_dir'][model_name][config.peft_method][dataset_name]
    
    # wait every GPU processes to reach here
    torch.distributed.barrier()  
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
        print(f"已创建新的权重存储路径: {save_path}") 
    
    # accelerator.save({  
    #     'prefix_encoder': model.prefix_encoder.state_dict(),  
    #     'classifier': model.classifier.state_dict()  
    # }, save_path)  
    
    accelerator.save(model.state_dict(), save_path)  
    



if __name__ == "__main__":
    '''
    
    '''
    model_path = Config["models"]["bert-base-uncased"]["model_path"]

    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)
    
    model_name = get_model_name_using_model(model)
    dataset_name = 'race'
    
    max_seq_length = get_max_length_from_model(model)
    
    config = PrefixTuningTrainerConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name=dataset_name,
        max_seq_length= max_seq_length  
    )
    train_prefix_tuning(config)