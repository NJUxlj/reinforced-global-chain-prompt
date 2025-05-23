import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW, Adam
from config.config import Config
import numpy as np
import evaluate
import os
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (   
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score,  
    classification_report  
)  
from typing import Tuple, List, Tuple, Optional, Union, Dict, Any



from data_utils.load import *

from utils import *


from torch.utils.data import DataLoader
from datasets import (
    Dataset, 
    load_dataset
)

from transformers import (
    RobertaModel,
    set_seed,
    default_data_collator,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup,
    PreTrainedModel
)
from peft import (
    TaskType,
    PeftType,
    PeftModel,
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

from swanlab.integration.accelerate import SwanLabTracker

from tqdm import tqdm

from evaluation.evaluation import ModelEvaluator



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
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 5
    num_epochs:int = 5
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                         # 最大序列长度  
    learning_rate: float = 5e-5                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768               # MLP中的P_theta'  即，MLP输入的隐单元维度  huggingface 默认它==encoder_hidden_size
    prefix_projection_hidden_size:int = 4*768  # 论文中重参数化用的MLP层的中间维度是hidden_size的4倍 
    encoder_hidden_size:int = 768  # 编码器的隐藏层大小
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减 
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # 用于AdaFactor optimizer
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = AdamW 
    
    seed:int=42
    train_size:int=22000
    mixed_precision:bool = False
    
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
    
    
    
class PrefixcTuningModelForSequenceClassification(nn.Module):
    def __init__(self, peft_model:PeftModel, seq_cls_model:AutoModelForSequenceClassification):
        super().__init__()
        self.peft_model = peft_model
        self.base_model:PreTrainedModel = peft_model.base_model
        self.seq_cls_model = seq_cls_model
        self.model_config=self.seq_cls_model.config
        self.config = self.seq_cls_model.config
        self.model_type =self.model_config.model_type
        
        self.classifier = get_classifier(self.seq_cls_model)
        
    
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor]=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions:bool=True,
        output_hidden_states:bool=True,
        return_dict:bool=True, # 返回 SequenceClassifierOutput 对像
        ):  
        
        device = input_ids.device  
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        
        # past_key_values = self.prefix_encoder(batch_size) # shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
        
        
        
        # # 准备前缀的注意力掩码  
        # prefix_attention_mask = torch.ones(  
        #     batch_size, 
        #     self.prefix_encoder.prefix_length, 
        #     device=attention_mask.device  
        # )  # shape = (batch_size, prefix_length)
        
        # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)   # shape = (batch_size, max_length), where prefix_length + input_length = max_length
        
        if self.model_type != 'gpt2':
            outputs = self.peft_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  # shape = (batch_size, max_length)
                token_type_ids=token_type_ids if self.model_type == "bert" else None,  
                position_ids=position_ids if self.model_type == "bert" else None,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds, # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是重新计算
                output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                return_dict=return_dict,
                # past_key_values=past_key_values, # shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
            )   # last_hidden_state, pooled_output = model(input_ids)
        else:
            outputs = self.peft_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  # shape = (batch_size, max_length)
                inputs_embeds=inputs_embeds, # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是重新计算
                output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                return_dict=return_dict,
            )   # last_hidden_state, pooled_output = model(input_ids)
        
        # pooled_output = outputs[1] # shape = (batch_size, hidden_size)
        
        if self.model_type=='qwen2':
            last_hidden_state = outputs.last_hidden_state  
            sequence_lengths = attention_mask.sum(dim=1) - 1  # 减1获取最后一个非padding位置  shape = (batch_size,)
            batch_size = input_ids.shape[0]  
            # 在每个样本中锁定最后那个非padding位置
            sequence_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]  # shape = (batch_size, hidden_size)
            cls_token = sequence_output
            # cls_token = self.dropout(cls_token)  # qwen2 的分类器自带dropout
                  
        elif self.model_type=='roberta':
            # RobertaClassificationHead 里面自己会自动取 [:,0,:]
            last_hidden_state = outputs.last_hidden_state
            cls_token = last_hidden_state
        elif self.model_type=='gpt2':
            # 使用最后一个非padding token的隐藏状态  
            last_hidden_state = outputs.last_hidden_state  
            cls_token = last_hidden_state
        else:
            # 类似于 Bert这样的
            cls_token = outputs.last_hidden_state[:, 0, :] # shape = (batch_size, hidden_size)
            # cls_token = self.dropout(cls_token)




        if self.model_type == 'gpt2':
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0] 

            logits= self.classifier(cls_token) # shape = (batch_size, seq_len, num_labels)
            
            # 处理填充token相关的逻辑  
            if self.model_config.pad_token_id is None and batch_size != 1:  
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")  
            if self.model_config.pad_token_id is None:  
                sequence_lengths = -1  
            else:  
                if input_ids is not None:  
                    # 找到每个序列中最后一个非填充token的位置  
                    sequence_lengths = torch.eq(input_ids, self.model_config.pad_token_id).int().argmax(-1) - 1  
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]  
                    sequence_lengths = sequence_lengths.to(logits.device)  
                else:  
                    sequence_lengths = -1  
            logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] # shape = (batch_size, num_labels)
        else:
            logits = self.classifier(cls_token) # shape = (batch_size, num_labels)
        # logits = logits.reshape(-1, config.num_labels)
        
        
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        if not return_dict: # 输出嵌套元组
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

def train_prefix_tuning(config:PrefixTuningTrainerConfig=None):
    fix_seed(config.seed)
    setup_distributed()
    # 初始化参数  
    model_name = config.model_name
    model, tokenizer = prepare_model_tokenizer(config.model_path, AutoModelForSequenceClassification, config.model_path, config.num_labels)
    
    # print("model.classifier.shape = ", model.classifier.shape)
    print_model_info(model)
    
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
    
    
    exp_name = f"{config.peft_method}_{model_name}_{dataset_name}"
    tracker = SwanLabTracker("PROMPT_TUNING_TRAING", experiment_name=exp_name)  # 训练可视化
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
    accelerator.init_trackers("PROMPT_TUNING_TRAING", config=tracker_config)
    
    wrapper = McqDatasetWrapper(
        model_name_or_path=config.model_path,
        max_seq_length=config.max_seq_length
    )
    
    dataset_configs = wrapper.dataset_configs
    dataset_config = dataset_configs[config.dataset_name]
    
    processed_ds = preprocess_dataset_peft(
        dataset_name, 
        model_path=model_path, 
        max_length=max_length,
        train_size=config.train_size,
        batch_size =config.batch_size,
        tokenizer=tokenizer
        )
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]  
    
    
    print("training set size = ", len(train_ds))
    print("eval set size = ", len(eval_ds))
    
    # 使用DistributedSampler进行数据分布  
    train_sampler = DistributedSampler(  
        train_ds,  
        shuffle=False,  
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
            batch_size=batch_size*dataset_config.num_options,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=batch_size*dataset_config.num_options,
            pin_memory=True,
            sampler = eval_sampler,
            drop_last=True,
        )
    
    
    
    # Prefix-tuning
    peft_config = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",
        inference_mode=False,
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=prefix_length, 
        token_dim = config.prefix_hidden_size,      
        # num_transformer_submodules=2,   # 论文中只对transformer block 中的 K, V 使用了prefix，所以是2而不是1  
        # num_attention_heads=12,
        # num_layers=12,
        encoder_hidden_size=config.encoder_hidden_size,  # bert隐藏层维度
        prefix_projection=config.prefix_projection, # # 论文中使用了MLP进行prefix投影
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    # 使用 get_peft_model 包装模型时，PEFT 库通常会自动冻结基础模型的参数，并将 PEFT 参数设置为可训练。
    base_model=None
    if hasattr(model, "base_model"):
        base_model = model.base_model
    elif hasattr(model, 'model'):
        base_model = model.model
    elif hasattr(model,'bert'):
        base_model = model.bert
    elif hasattr(model,'roberta'):
        base_model = model.roberta
    
    peft_model = get_peft_model(base_model, peft_config)
    
    verify_peft_config(peft_model, peft_config, config)
    
    # make sure to frozen the base model parameters
    for param in peft_model.base_model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix tokens is trainable
    for name, param in peft_model.named_parameters():  
        if 'prefix_encoder' in name:  
            param.requires_grad = True 
    
    peft_model.print_trainable_parameters()
    
    model = PrefixcTuningModelForSequenceClassification(peft_model, model)
    
    
    model.to(accelerator.device)
    
    


    # make sure that the fine-tuning will only update virual tokens
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)
    
    
    if accelerator.is_main_process:
        # logging_dir = Config['logging_dir'][model_name][config.peft_method][dataset_name]
        logging_dir = f'./logs/{model_name}/{config.peft_method}/{dataset_name}/'
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name="log_eval", logging_dir=logging_dir, log_level="INFO")
    
    
    if accelerator.is_main_process:  
        accelerator.init_trackers("training")
    
    global_step = 0
    optimizer.zero_grad()
    param_monitor = {}
    evaluator = ModelEvaluator(accelerator,dataset_config)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if train_sampler is not None:  # 每轮训练都打乱顺序
            train_sampler.set_epoch(epoch) 
        
        # 记录每个epoch开始时的参数状态  
        if accelerator.is_main_process:
            record_epoch_param_state(model, param_monitor)
        
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
                print_prediction_distribution(outputs,step,loss, dataset_config.num_options, logits, labels)

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
        # eval_results = evaluate_prefix_tuning(model, eval_dataloader, accelerator)
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
            


    # 保存权重
    print("model name = ", model_name)
    # save_path = Config['save_model_dir'][model_name][config.peft_method][dataset_name]
    

def evaluate_prefix_tuning(
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



if __name__ == "__main__":
    '''
    
    '''
    
    args = parse_training_arguments()
    dataset_name =args.dataset_name
    model_name = args.model_name
    model_path = get_model_path_by_model_name(model_name)
    
    # model_name = get_model_name_using_model(model)
    # dataset_name = 'race'
    # model_path = Config["models"]["roberta-large"]["model_path"]

    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels=2)
    

    
    max_seq_length = get_max_length_from_model(model)
    
    hidden_size = get_hidden_size_using_model(model)
    
    config = PrefixTuningTrainerConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name=dataset_name,
        max_seq_length= max_seq_length,
        num_epochs=args.num_epochs,
        num_labels=2,
        prefix_length=100,
        prefix_hidden_size=hidden_size,
        prefix_projection_hidden_size=4*hidden_size,
        encoder_hidden_size=hidden_size,
        train_size=args.train_size,
        batch_size=args.batch_size,
    )
    train_prefix_tuning(config)