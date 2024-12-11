import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW
import numpy as np
import argparse
import evaluate
import os
from config import Config
from dataclasses import dataclass


from typing import Tuple, List, Tuple, Optional, Union, Dict, Any

from load import *

from utils import *


from torch.utils.data import (
    DataLoader,
    Dataset
)
from datasets import (
    #Dataset, 
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

from transformers.modeling_outputs import (
    SequenceClassifierOutput, # 是一种官方的输出格式
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput
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
    Accelerator
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


# we follow the setting of prompt-tuning (Lester et al., 2021)
@dataclass  
class PtuningV2Config:  
    """MCQA任务的P-tuning V2配置"""  
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "p-tuning-v2"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 100                        # 前缀长度  
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 5
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                     # 最大序列长度  
    learning_rate: float = 1e-3                   # 前缀参数的学习率  
    classifier_learning_rate:float=1e-4
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768                 # 前缀投影隐藏层大小  
    warmup_steps: int = 500  # 添加预热步骤  
    warmup_ratio:float=0.06
    weight_decay: float = 0.01 #1e-5 # 添加权重衰减  
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.999   # beta2: 二阶矩估计的指数衰减率（默认0.999）
    adam_epsilon:float = 1e-8 
    total_training_steps = 22000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = AdamW
    
    seed:int=42
    train_size:int=22000
    mixed_precision:bool=False

class PrefixEncoder(nn.Module):  
    """前缀编码器"""  
    def __init__(self, config: PtuningV2Config, model_config:AutoConfig, device = Config.device):  
        super().__init__()  
        self.device = device
        self.prefix_length = config.prefix_length  
        self.hidden_size = model_config.hidden_size  
        self.n_layer = model_config.num_hidden_layers  
        self.n_head = model_config.num_attention_heads  
        self.prefix_projection = config.prefix_projection  
        
        if self.prefix_projection:  
            # 使用MLP来生成前缀  
            self.embedding = nn.Embedding(self.prefix_length, config.prefix_hidden_size).to(self.device)  
            self.trans = nn.Sequential(  
                nn.Linear(config.prefix_hidden_size, config.prefix_hidden_size),  
                nn.Tanh(),  
                nn.Linear(config.prefix_hidden_size, self.n_layer * 2 * self.hidden_size)  
            ).to(self.device)  
        else:  
            # 直接优化前缀参数  
            '''
            self.n_layer * 2  是因为要把prefix embedding插入到每一个transformer层的W_k, W_v矩阵上
            '''
            self.embedding = nn.Embedding(self.prefix_length, self.n_layer * 2 * self.hidden_size).to(self.device) 

    def forward(self, batch_size: int) -> torch.Tensor:  
        '''
         # 返回类型：Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  
            # 长度为num_layers的元组，每个元素是(key, value)对  
        
        '''
        prefix_tokens = torch.arange(self.prefix_length).long()   # shape = (prefix_length)
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1)   # shape = (batch_size, prefix_length)
        
        if prefix_tokens.device != self.embedding.weight.device:  
            prefix_tokens = prefix_tokens.to(self.embedding.weight.device)  
            
        if self.prefix_projection:  
            prefix_embeddings = self.embedding(prefix_tokens)  
            past_key_values = self.trans(prefix_embeddings)  
        else:  
            past_key_values = self.embedding(prefix_tokens)  
            
        # # [batch_size, prefix_length, n_layer * 2 * hidden_size]  
        
        # 重塑维度为每层的key和value  
        past_key_values = past_key_values.view(  
            batch_size,  
            self.prefix_length,  
            self.n_layer * 2,  
            self.n_head,  
            self.hidden_size // self.n_head  
        )  # # [batch_size, prefix_length, n_layer * 2, n_head, head_dim]  
        
        
        
        # 转置为符合注意力机制的格式  
        past_key_values = past_key_values.permute(2, 0, 3, 1, 4)  # shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
        # 分离key和value  
        
        '''
            沿着第一个维度每2个切分一次返回一个包含num_layers个元组的元组，
            每个内部元组包含两个张量(key和value)。
            
            
            # 返回类型：Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  
            
            # 每个张量形状: [batch_size, n_head, prefix_length, head_dim]  
            
            # 长度为num_layers的元组，每个元素是(key, value)对  
            (  
                # Layer 0  
                (    
                    key_layer0,   # shape: [batch_size, num_heads, prefix_length, head_dim]  
                    value_layer0  # shape: [batch_size, num_heads, prefix_length, head_dim]  
                ),  
                # Layer 1  
                (  
                    key_layer1,   # shape: [batch_size, num_heads, prefix_length, head_dim]  
                    value_layer1  # shape: [batch_size, num_heads, prefix_length, head_dim]  
                ),  
                # ... 更多层  
            )  
        
        '''
        return past_key_values.split(2)  # 


# # 定义Prompt Encoder，用于生成可训练的提示向量  
# class PromptEncoder(nn.Module):  
#     def __init__(self, prompt_length, embedding_size, hidden_size):  
#         super(PromptEncoder, self).__init__()  
#         self.prompt_length = prompt_length  
#         # 初始化可训练的提示embedding  
#         self.prompt_embeddings = nn.Embedding(prompt_length, embedding_size)  
#         # 将提示embedding映射到模型的hidden size  
#         self.transform = nn.Sequential(  
#             nn.Linear(embedding_size, hidden_size),  
#             nn.Tanh(),  
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),  
#         )  

#     def forward(self, batch_size, device):  
#         # 创建提示的索引  
#         prompt_indices = torch.arange(self.prompt_length).long().to(device)  
#         # 获取提示embedding  
#         prompt_embeddings = self.prompt_embeddings(prompt_indices)  
#         # 通过transform层  
#         prompt_embeddings = self.transform(prompt_embeddings)  
#         # 扩展batch维度  
#         prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
#         return prompt_embeddings 


 

# 定义P-Tuning V2的BERT模型  
class PTuningV2ForSequenceClassification(nn.Module):  
    def __init__(self, model, num_labels, prompt_length, config:PtuningV2Config=None, device=Config.device):  
        super(PTuningV2ForSequenceClassification, self).__init__()  
        self.config = config
        self.device = device
        # 加载预训练的BERT模型  
        # 加载预训练模型  
        self.model = config.auto_model_class.from_pretrained(  
            config.model_path,  
            num_labels=config.num_labels  
        ) 
        self.model.to(self.device) 
        self.model_config = self.model.config
        
        # 获取模型类型  
        self.model_type = self.model_config.model_type  
        
        # 获取基础模型  
        self.base_model = self.model.base_model  
        self.base_model.to(self.device)
        
        # 冻结基础模型参数  (这里不能用self.model, 因为这样会把classifier也冻结了)
        for param in self.base_model.parameters():  
            param.requires_grad = False   
        
        # 初始化前缀编码器  
        self.prefix_encoder = PrefixEncoder(config, self.model_config, device=self.device)  
        
        self.dropout = torch.nn.Dropout(self.model_config.hidden_dropout_prob).to(self.device)
        
        # self.prefix_tokens = torch.arange(self.prefix_encoder.prefix_length).long() 
        
        
        self.classifier:nn.Module = get_classifier_from_model(self.model).to(self.device)
        
        # for param in self.classifier.parameters():  
        #     param.requires_grad = False 
        
        self._freeze_parameters()
        
        self.print_total_params()
        
    def get_prompt(self, batch_size):  
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)  
        past_key_values = self.prefix_encoder(prefix_tokens)  
        return past_key_values  
    
    def _freeze_parameters(self):  
        # 冻结所有参数  
        for param in self.base_model.parameters():  
            param.requires_grad = False  
            
        for param in self.prefix_encoder.parameters():  
            param.requires_grad = True  
        
        for param in self.classifier.parameters():  
            param.requires_grad = True 
            

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
        batch_size = input_ids.size(0)  
        
        past_key_values = self.prefix_encoder(batch_size) # shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
        
        
        
        # 准备前缀的注意力掩码  
        prefix_attention_mask = torch.ones(  
            batch_size, 
            self.prefix_encoder.prefix_length, 
            device=attention_mask.device  
        )  # shape = (batch_size, prefix_length)
        
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)   # shape = (batch_size, max_length), where prefix_length + input_length = max_length
        
        
        outputs = self.base_model(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids if self.model_type == "bert" else None,  
            position_ids=position_ids if self.model_type == "bert" else None,
            # labels=labels,  
            head_mask=head_mask,
            inputs_embeds=inputs_embeds, # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是重新计算
            output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
            output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
            return_dict=return_dict,
            past_key_values=past_key_values, # shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
        )   # last_hidden_state, pooled_output = model(input_ids)
        
        # pooled_output = outputs[1] # shape = (batch_size, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0, :] # shape = (batch_size, hidden_size)

        cls_token = self.dropout(cls_token)
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
        
        # return outputs
        
        
        # # 获取可训练的提示embedding  
        # prompt_embeddings = self.prompt_encoder(batch_size, device)  
        # # 获取原始的输入embedding  
        # inputs_embeds = self.base_model.embeddings(input_ids)  
        # # 将提示embedding与输入embedding拼接  
        # inputs_embeds = torch.cat([prompt_embeddings, inputs_embeds], dim=1)  
        # # 调整attention mask  
        # prompt_attention_mask = torch.ones(batch_size, self.prompt_length).to(device)  
        # attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)  
        # # 通过BERT模型  
        # # outputs.shape = (batch_size, max_length, hidden_size)
        # outputs = self.base_model(  
        #     inputs_embeds=inputs_embeds,  
        #     attention_mask=attention_mask  
        # )  
        # # 获取[CLS]位置的输出  
        # # outputs.last_hidden_state.shape = (batch_size, max_length, hidden_size)
        # pooled_output = outputs.last_hidden_state[:, 0]  
        # # 通过分类器  
        # logits = self.classifier(pooled_output)  
        # loss = None  
        # if labels is not None:  
        #     # 计算损失  
        #     loss_fn = nn.CrossEntropyLoss()  
        #     loss = loss_fn(logits, labels)  
        # return {'loss': loss, 'logits': logits}  
    
    
    
    
    def print_total_params(self):
        bert_param = 0
        for name, param in self.base_model.named_parameters():
            bert_param += param.numel()
            
        classifier_param = 0
        for name, param in self.classifier.named_parameters():
            classifier_param += param.numel()
            print('classifier weight:{}, trainable number is {}'.format(name, param.numel()))
            
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            
        for name,param in self.prefix_encoder.named_parameters():
            print('prefix encoder weight:{}, trainable number is {}'.format(name, param.numel())) 
            
        total_param = all_param - bert_param - classifier_param
        print('total trainable param is {}'.format(total_param)) # 9860105
        print('total trainable param percentage is {}%'.format((total_param / all_param)*100)) # 9860105
        



# 定义数据集类  
class Racedataset(Dataset):  
    def __init__(self, dataset_split, tokenizer, prompt_length, max_length=512):  
        self.input_ids = []  
        self.attention_masks = []  
        self.labels = []  
        self.prompt_length = prompt_length  
        self.max_length = max_length - prompt_length  # 调整最大长度，考虑提示长度  
        for item in dataset_split:  
            article = item['article']  
            question = item['question']  
            options = item['options']  
            answer = item['answer']  

            # 获取正确答案的索引  
            label_idx = ord(answer) - ord('A')  # RACE的答案是'A', 'B', 'C', 'D'  

            for i, option in enumerate(options):  
                # 构建输入文本  
                input_text = f"{article} [SEP] 问题：{question} 选项：{option}"  
                # 对输入文本进行编码  
                encoding = tokenizer(  
                    input_text,  
                    max_length=self.max_length,  
                    padding='max_length',  
                    truncation=True,  
                    return_tensors='pt'  
                )  
                self.input_ids.append(encoding['input_ids'].squeeze())  
                self.attention_masks.append(encoding['attention_mask'].squeeze())  
                # 标签：正确选项为1，其他为0  
                self.labels.append(1 if i == label_idx else 0)  

    def __len__(self):  
        return len(self.labels)  

    def __getitem__(self, idx):  
        return {  
            'input_ids': self.input_ids[idx],  
            'attention_mask': self.attention_masks[idx],  
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  
        }  
        
        
def train_p_tuning_v2(config: PtuningV2Config=None):
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
    tracker = SwanLabTracker("P_TUNING_V2_TRAING", experiment_name=exp_name)  # 训练可视化
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
    accelerator.init_trackers("P_TUNING_V2_TRAING", config=tracker_config)
    
    
    
    processed_ds = preprocess_dataset_peft(dataset_name, model_path=config.model_path, max_length=max_length)
    
    
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
            pin_memory=True,
            sampler=train_sampler
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            pin_memory=True,
            sampler=eval_sampler
        )

    # encapsulate the base model into a P-Tuning V2 model
    model = PTuningV2ForSequenceClassification(
            model, 
            num_labels, 
            prefix_length, 
            config=config, 
            device=accelerator.device
        )  
    model.to(accelerator.device)  
    
    model.print_total_params()
    
    if accelerator.is_main_process:    
        # 打印可训练参数  
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        print(f"Trainable parameters (self-calculated): {trainable_params}")  


    # 只优化Prompt Encoder和分类器的参数  
    # optimizer = optim.Adam([  
    #     {'params': model.prompt_encoder.parameters()},  
    #     {'params': model.classifier.parameters()}  
    # ], lr=learning_rate)  
    
    # 优化器现在需要同时优化前缀编码器和分类器的参数  
    optimizer = torch.optim.AdamW([  
        {
            'params': model.prefix_encoder.parameters(), 
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay
        },  
        {
            'params': model.classifier.parameters(), 
            'lr': config.classifier_learning_rate,
            'weight_decay': config.weight_decay  
        }  
    ], weight_decay=config.weight_decay, betas=(config.beta1_decay, config.beta2_decay),eps=config.adam_epsilon) 
    
    
    num_training_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)  
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)
    
    model:PTuningV2ForSequenceClassification
    
    print("type(model) = ", type(model))
    
    if accelerator.is_main_process:
        # logging_dir = Config['logging_dir'][model_name][config.peft_method][dataset_name]
        logging_dir = f'./logs/{model_name}/{config.peft_method}/{dataset_name}/'
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name="log_eval", logging_dir=logging_dir, log_level="INFO")
      
    if accelerator.is_main_process:  
        accelerator.init_trackers("training")
        

    # 训练循环  
    global_step = 0 
    optimizer.zero_grad()
    # 添加参数状态监控  
    param_monitor = {}
    
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
            
            logits = outputs.logits
            loss = outputs.loss
            # loss = criterion(logits, labels.long())
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
        eval_results = evaluate_ptuning_v2(model, eval_dataloader, accelerator)
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
    
    # print("model name = ", model_name)
    # save_path = Config[SAVE_DIR][model_name][config.peft_method][dataset_name]
    # unwrapped_model = accelerator.unwrap_model(model)
    
    # # wait every GPU processes to reach here
    # torch.distributed.barrier()  
    
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))  
    #     print(f"已创建新的权重存储路径: {os.path.dirname(save_path)}") 
    
    # 只保存prompt embedding  
    # prompt_state_dict = {}  
    # for name, param in unwrapped_model.named_parameters():  
    #     if 'prefix_encoder' in name:  
    #         prompt_state_dict[name] = param.data.cpu()
    
    # unwrapped_model.save_pretrained(  
    #     save_path,  
    #     state_dict=prompt_state_dict,  
    # )  
    
    # accelerator.save({  
    #     'prefix_encoder': model.prefix_encoder.state_dict(),  
    #     'classifier': model.classifier.state_dict()  
    # }, save_path)  
    
def evaluate_ptuning_v2(
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

def evaluate_ptuning_v2_legacy(model, eval_dataloader, accelerator:Accelerator):  
    """  
    评估函数  
    """  
    model.eval()  
    all_preds = []  
    all_labels = []  
    
    with torch.no_grad():  
        for batch in eval_dataloader:  
            outputs = model(**batch)  
            logits = outputs.logits  
            labels = batch['labels']  
            
            # 使用accelerator.gather收集所有进程的预测结果  
            gathered_logits = accelerator.gather(logits)  
            gathered_labels = accelerator.gather(labels)  
            
            preds = torch.argmax(gathered_logits, dim=1).cpu().numpy()  
            labels = gathered_labels.cpu().numpy()  
            
            all_preds.extend(preds)  
            all_labels.extend(labels)  
    
    # 计算评价指标  
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
    precision, recall, f1, _ = precision_recall_fscore_support(  
        all_labels, all_preds, average='weighted', zero_division=0  
    )  
    
    return {  
        'accuracy': accuracy,  
        'precision': precision,  
        'recall': recall,  
        'f1': f1  
    }


if __name__ == "__main__":
    '''
    
    '''
    
    
    # model_name = "bert-base-uncased"
    # dataset_name = "race"
    # model_path = Config["models"]["bert-base-uncased"]["model_path"]
    # hidden_size = Config["models"]["bert-base-uncased"]["hidden_dim"]

    args = parse_training_arguments()
    dataset_name =args.dataset_name
    model_name = args.model_name
    model_path = get_model_path_by_model_name(model_name)
    
    model, tokenizer = prepare_model_tokenizer(model_path,AutoModelForSequenceClassification, model_path)

    max_seq_length = get_max_length_from_model(model)
    hidden_size = get_hidden_size_using_model(model)

    config = PtuningV2Config(
        model_name = model_name,
        model_path=model_path,
        auto_model_class = AutoModelForSequenceClassification,
        dataset_name=dataset_name,
        # learning_rate=Config['learning_rate'],
        max_seq_length=max_seq_length,
        num_labels=2,
        # batch_size=Config['batch_size'],
        # num_epochs = Config['num_epochs'],
        prefix_projection=True,
        prefix_hidden_size=hidden_size,
        prefix_length=100,
        train_size=args.train_size,
        batch_size=args.batch_size,
        
        
    )

    # 这里传model进去只是为了符合其他fine-tuning的写法，实际上这里不需要model
    train_p_tuning_v2(config)
    
    
    