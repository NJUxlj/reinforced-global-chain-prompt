
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
from swanlab.integration.accelerate import SwanLabTracker

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (   
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score,  
    classification_report  
)

from evaluation import ModelEvaluator

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
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 5
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                     # 最大序列长度  
    learning_rate: float = 0.3                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768    
    encoder_hidden_size:int = 768  # 重参数化器MLP的隐藏层大小# prefix token 的维度 (P_theta')
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减  
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # beta2: 二阶矩估计的指数衰减率（默认0.999）
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = Adam
    
    seed:int=42
    train_size:int=22000
    mixed_precision:bool=False


def train_p_tuning(config:PtuningConfig):
    fix_seed(config.seed)
    setup_distributed()
    print("\n\n",config,"\n\n")
    model_name = config.model_name
    model, tokenizer = prepare_model_tokenizer(config.model_path, AutoModelForSequenceClassification, config.model_path )
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
    dataset_name = config.dataset_name
    dataset_path = get_dataset_path_by_name(dataset_name)
    
    exp_name = f"{config.peft_method}_{model_name}_{dataset_name}"
    tracker = SwanLabTracker("P_TUNING_TRAING", experiment_name=exp_name)  # 训练可视化
    accelerator = Accelerator(
        # gradient_accumulation_steps=500,  
        # mixed_precision=config.mixed_precision,
        log_with=[tracker]
    )
    tracker_config = {
        "num_epoch": config.num_epochs,
        "batch_num": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }
    accelerator.init_trackers("P_TUNING_TRAING", config=tracker_config)
    wrapper = McqDatasetWrapper(
        model_name_or_path=config.model_path,
        max_seq_length=config.max_seq_length
    )
    
    dataset_configs = wrapper.dataset_configs
    dataset_config = dataset_configs[config.dataset_name]
    
    processed_ds = preprocess_dataset_peft(dataset_name, model_path=config.model_path, max_length=max_length, train_size=config.train_size)
    
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]

    print("training set size = ", len(train_ds))
    print("eval set size = ", len(eval_ds))
    
    train_sampler = DistributedSampler(  
        train_ds,  
        shuffle=True,  
        seed=42  
    ) if torch.distributed.is_initialized() else None 
    
    eval_sampler = DistributedSampler(  
        eval_ds,  
        shuffle=False,  
        seed=42  
    ) if torch.distributed.is_initialized() else None 
    
    train_dataloader = DataLoader(
            train_ds, 
            # shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            pin_memory=False,
            sampler=train_sampler
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            pin_memory=False,
            sampler = eval_sampler
        )
    
    # Prompt-tuning
    peft_config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type= TaskType.SEQ_CLS, 
        num_virtual_tokens=prefix_length, 
        token_dim=config.prefix_hidden_size, # 与基础模型bert保持一致
        # num_transformer_submodules=1,
        # num_attention_heads=12,
        # num_layers=12,
        encoder_reparameterization_type="LSTM",
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_num_layers=2,
        encoder_dropout= 0.1,
        inference_mode=False   # 训练模式
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    model = get_peft_model(model, peft_config)
    model.to(accelerator.device)
    
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
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)

    print("type(model) = ", type(model))
    
    if accelerator.is_main_process:
        logging_dir = f'./logs/{model_name}/{config.peft_method}/{dataset_name}/'
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name="log_eval", logging_dir=logging_dir, log_level="INFO")
    
    if accelerator.is_main_process:  
        accelerator.init_trackers("training")
    
    
    global_step = 0

    optimizer.zero_grad()
    # 添加参数状态监控  
    param_monitor = {}

    evaluator = ModelEvaluator(accelerator, dataset_config)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if train_sampler is not None:  # 每轮训练都打乱顺序
            train_sampler.set_epoch(epoch) 
            
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            labels = batch["labels"]  
            outputs = model(**batch)
            criterion = nn.CrossEntropyLoss()
            
            logits = outputs.logits
            # 可以添加对比损失  
            # ---------------------------------
            logits = logits.view(-1, dataset_config.num_options, 2)[:, :, 1]  # [batch_size, 4, 2] -> [batch_size, 4]  
            labels = labels.view(-1, dataset_config.num_options).argmax(dim=1)  # [batch_size, 4] -> [batch_size, ]
            # ---------------------------------
            loss = criterion(logits, labels.long())
            if step % 300 == 0 and step!=0 and accelerator.is_main_process:  
                # 打印训练过程中的预测分布
                print_prediction_distribution(outputs,step,loss)
                
            total_loss += loss.detach().float().item() 

            
            accelerator.wait_for_everyone()
            accelerator.backward(loss, retain_graph=True)
            
            should_update = accelerator.sync_gradients
            
            if should_update:  
                if step % 300 == 0 and step!=0 and accelerator.is_main_process:
                    # 确保梯度确实在更新  
                    detect_param_grad_updates(model,epoch,step)
                    monitor_gradients(model, step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            del outputs
            del loss
            
            torch.cuda.empty_cache()
        global_step+=1
        accelerator.wait_for_everyone()
        
        # 在每个epoch结束后检查参数变化  
        if accelerator.is_main_process and epoch > 0:  
            check_param_after_epoch(model,param_monitor,epoch)
        
        if accelerator.is_main_process:
            print(f"begin epoch {epoch} evaluating...")   
        # eval_results = evaluate_p_tuning(model, eval_dataloader, accelerator)
        eval_results = evaluator.evaluate(model, eval_dataloader)
        accelerator.wait_for_everyone()
        # 记录自定义的logger
        if accelerator.is_main_process and logger is not None:
            avg_loss = total_loss / len(train_dataloader)   
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            logger.info({
                'epoch': epoch, 
                'avg_loss': avg_loss, 
                **eval_results
                })  
        
            # 记录到swanlab的logger， 用于可视化
            accelerator.log(
                {
                    'epoch': epoch, 
                    # 'avg_loss': avg_loss,
                    **eval_results
                }
            )

        model.train()
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        finish_words= f'A training for {model_name}_{config.peft_method}_{dataset_name} is done.'
        print("********************** ", finish_words, " ***************************")
        print("*******************************************************************")
        print("*******************************************************************")

    try:
        accelerator.end_training()
    except:
        print("****************************** End Training *******************************************")
        print("****************************** Clean Cuda cache *******************************************")
        print("****************************** destroy process group *******************************************")
        
        torch.cuda.empty_cache()  
        if torch.distributed.is_initialized():  
            torch.distributed.destroy_process_group() 
            
    
    # 保存模型
    
    print("model name = ", model_name)
    

    # save_path = Config['save_model_dir'][model_name]['p-tuning'][dataset_name]
    
    # # wait every GPU processes to reach here
    # torch.distributed.barrier()  
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)  
    #     print(f"已创建新的权重存储路径: {save_path}") 
    
    # # if accelerator.is_main_process():
    # #     torch.save()

    # accelerator.save(model.state_dict(), save_path)  
    
    
    
    
    
    
    
    
def evaluate_p_tuning(
        model, 
        eval_dataloader, 
        accelerator:Accelerator, 
    )->Dict:  
    """  
    评估函数  
    """  
    # 如果需要保持原始模型状态不变  
    # evaluation_mode = model.training  # 保存当前状态  
    
    # 保存原始训练状态  
    training_state = model.training  
    model.eval()  
    all_preds = []  
    all_labels = []  
    
    for batch in eval_dataloader:  
        with torch.no_grad():  

            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1) 
            labels = batch['labels']
            
            preds, labels = accelerator.gather_for_metrics(
                (preds, labels)
            )
            
        
            all_preds.append(preds.cpu())  
            all_labels.append(labels.cpu())   
            
    
    all_preds = torch.cat(all_preds).numpy()  
    all_labels = torch.cat(all_labels).numpy() 
            
    
    
    # 在主进程上计算指标  
    metrics = None   
    
    # 计算评价指标  
    metrics = {  
        'accuracy': accuracy_score(all_labels, all_preds),  
        'precision': precision_score(all_labels, all_preds, average='weighted'),  
        'recall': recall_score(all_labels, all_preds, average='weighted'),  
        'f1': f1_score(all_labels, all_preds, average='weighted')  
    }     
    
    accelerator.wait_for_everyone()     
    
    if accelerator.is_main_process: 
        
        # debug info
        print("\n****************** Evaluation Results:**************************")  
        print(f"Total samples evaluated: {len(all_labels)}")  
        print(f"Batch predictions distribution: {np.bincount(all_preds)}")  
        print(f"Batch labels distribution: {np.bincount(all_labels)}")   
        print("\n******************** Classification Report: ***********************")  
        print(classification_report(all_labels, all_preds))         
        print("*****************************************************************\n")
        
        
    
    # 恢复模型原始状态  
    model.train(training_state) 
        
    return metrics 
        
    
    
    

if __name__ == '__main__':
    '''
    
    '''
    # model_path = Config["models"]["bert-base-uncased"]["model_path"]
    # model_name = "bert-base-uncased"
    
    # dataset_name = "race"
    
    args = parse_training_arguments()
    dataset_name =args.dataset_name
    model_name = args.model_name
    train_size = args.train_size
    batch_size = args.batch_size
    model_path = get_model_path_by_model_name(model_name)

    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path)

    max_seq_length = get_max_length_from_model(model)
    hidden_size = get_hidden_size_using_model(model)
    
    config = PtuningConfig(
        model_name = model_name,
        model_path = model_path,
        max_seq_length=max_seq_length,
        dataset_name= dataset_name,
        num_epochs=args.num_epochs,
        num_labels=2,
        prefix_hidden_size=hidden_size,
        encoder_hidden_size=hidden_size,
        prefix_length=100,
        train_size = train_size,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
        
    )
    
    train_p_tuning(config)
    