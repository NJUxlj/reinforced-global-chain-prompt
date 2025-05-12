from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    AutoModel,
    AutoConfig,
    AutoTokenizer,  
    BertTokenizerFast,
    AutoModelForSequenceClassification,  
    Trainer,  
    TrainingArguments  
)

from transformers.modeling_outputs import (
    SequenceClassifierOutput, # 是一种官方的输出格式
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput
)
# from transformers import PromptForSequenceClassification  
from peft import (
    PromptTuningInit, 
    PromptTuningConfig,
    get_peft_model,
    TaskType
)

from accelerate import(
    Accelerator,
)
from accelerate.logging import get_logger


from data_utils.load import *

from utils import *

from models.causal_modeling import *

from src.models.rgc_components import (
    SentenceEncoder,
    BaasAttention
)

import time
import sys  
import os  
from torchsummary import summary
# 添加项目根目录到 Python 路径  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from autocot.make_embeddings import (
    get_cot_context,
    rollback_one_step_extend,
    ChainEncodingArguments,
    generate_new_step
)

from datasets import (
    load_dataset,
)
from torch.utils.data import (
    DataLoader,
    Dataset
)
from torch.utils.data.distributed import DistributedSampler

from config.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.optim import Adam
from copy import deepcopy
import numpy as np
import csv
import evaluate
import gensim
from gensim import corpora, models
import nltk
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (   
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score,  
    classification_report  
)  
from sklearn.metrics.pairwise import cosine_similarity  

from sklearn.cluster import KMeans
from typing import List, Dict
from collections import defaultdict

from sentence_transformers import SentenceTransformer, models
from dataclasses import dataclass
from collections import Counter
from swanlab.integration.accelerate import SwanLabTracker

from evaluation.evaluation import ModelEvaluator


import nltk  
# nltk.download('punkt') 
# nltk.download('punkt_tab') 
# nltk.download('stopwords')  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 




# device = Config['device']






def train_baas_prompt(config:BaasPromptConfig, chain_encode_args:ChainEncodingArguments):
    check_cuda_setup()
    setup_distributed()
    
    fix_seed(config.seed)
    print("\n\n",config,"\n\n")
    model_name = config.model_name
  
    model_path = config.model_path
    num_labels = config.num_labels
    batch_size = config.batch_size
    
    model,tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, num_labels=num_labels)

    # 定义双向Prompt Tuning的参数       
    num_prefix_tokens = config.prefix_length   # 前缀Prompt Tokens的数量  
    num_suffix_tokens = config.suffix_length   # 后缀Prompt Tokens的数量  

    lr = config.learning_rate
    num_epochs = config.num_epochs
    
    max_length = get_max_length_from_model(model)
    print(f"before inserting prompt tokens, {model_name}'s max length = {max_length}")
    max_length = max_length - num_prefix_tokens - num_suffix_tokens
    print(f"After inserting prompt tokens, {model_name}'s max length = {max_length}")

    
    dataset_name = config.dataset_name
    dataset_path = get_dataset_path_by_name(dataset_name)
    
    exp_name = f"{config.peft_method}_{model_name}_{dataset_name}"
    tracker = SwanLabTracker("BAAS_PROMPT_TRAING", experiment_name=exp_name)  # 训练可视化
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,  
        mixed_precision=config.mixed_precision,
        log_with=[tracker]
    )
    tracker_config = {
        "num_epoch": config.num_epochs,
        "batch_num": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }
    accelerator.init_trackers("BAAS_PROMPT_TRAING", config=tracker_config)
    
    wrapper = McqDatasetWrapper(
        model_name_or_path=config.model_path,
        max_seq_length=config.max_seq_length
    )
    
    dataset_configs = wrapper.dataset_configs
    dataset_config = dataset_configs[config.dataset_name]   
    
    processed_ds = preprocess_dataset_peft(
        dataset_name, 
        model_path, 
        max_length = max_length, 
        seq_cls_type=config.seq_cls_type,
        train_size=config.train_size,
        batch_size =config.batch_size,
        tokenizer = tokenizer,
        )
    
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]
    print("************************* Dataset Size *****************************")
    print(f"Train set size: {len(train_ds)}")  
    print(f"Validation set size: {len(eval_ds)}")     
    print("***************************************************************\n")

    print("dataset is preprocessed successfully ~~~")
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
    
    
    data_collator=default_data_collator
    # if model.config.model_type=='qwen2':
    #     data_collator = DataCollatorWithPadding(  
    #         tokenizer=tokenizer,  
    #         padding=True,  
    #         return_tensors="pt"  
    #     )  
    # else:
    #     data_collator = default_data_collator
    
    train_dataloader = DataLoader(
            train_ds, 
            # shuffle=True, # shuffle is not necessary when using DistributedSampler
            collate_fn=data_collator, 
            batch_size=batch_size*dataset_config.num_options,
            # pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=data_collator, 
            batch_size=batch_size*dataset_config.num_options,
            # pin_memory=True,
            sampler=eval_sampler,
            drop_last=True,
        )
    
    
    baas_model = BassPromptModel(  
        model=model,
        tokenizer=tokenizer,
        config=config,
        chain_encode_args=chain_encode_args,
        device = accelerator.device,
        debug=config.debug,
        num_options=dataset_config.num_options
    )
    baas_model.to(accelerator.device)
    
    gram = 0 # each float = 4B
    for p in baas_model.parameters():
        gram += p.numel()
    gram = gram * 4 / 1024 / 1024

    print(f"=== gram: {gram} ===")
    # return
    
    optimizer = config.optimizer_class(
        filter(lambda p: p.requires_grad, baas_model.parameters()), 
        lr=lr
    )
    
    # 计算总训练步数  
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)  
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        # num_training_steps=(len(train_dataloader) * num_epochs),
        num_training_steps=max_train_steps  
    )
    

    model = baas_model 
    
    
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader= accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)
    
    if accelerator.is_main_process:
        # logging_dir = Config['logging_dir'][model_name][config.peft_method][dataset_name]
        logging_dir = f'./logs/{model_name}/{config.peft_method}/{dataset_name}/'
        logger_name = f"{model_name}_{config.peft_method}_{dataset_name}_{config.seq_cls_type}_{config.classes_initiate_method}_suffix_ratio_{config.suffix_ratio}"
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name=logger_name, logging_dir=logging_dir, log_level="INFO")

    if accelerator.is_main_process:  
        accelerator.init_trackers("training") 
        
    global_step = 0
    optimizer.zero_grad()
    param_monitor = {}
    evaluator = ModelEvaluator(accelerator,dataset_config)
    print("\n\n**************************************************************************************")
    print(f"************* Start training model {model_name} on {dataset_name} using {config.peft_method} ... ************")
    print(f"****************** Prefix topic labels retrived by {config.classes_initiate_method} ************\n\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if train_sampler is not None:  # 每轮训练都打乱顺序
            train_sampler.set_epoch(epoch) 
        # 记录每个epoch开始时的参数状态  
        if accelerator.is_main_process:
            record_epoch_param_state(model, param_monitor)
        
        for step, batch in enumerate(tqdm(train_dataloader)):

            with accelerator.accumulate(model):
                if step % 1000 == 0:  
                    if accelerator.is_main_process:
                        # 确保梯度确实在更新  
                        detect_param_grad_updates(model,epoch,step)
                        monitor_gradients(model, step)
                    # 定期清理缓存
                    torch.cuda.empty_cache()    
                
                
                
                labels = batch["labels"]  
                
                outputs = model(**batch)
                
                criterion = nn.CrossEntropyLoss()
                
                logits = outputs.logits
                
                # 可以添加对比损失  
                # ---------------------------------
                logits = logits.view(-1, dataset_config.num_options, 2)[:, :, 1]  # [batch_size, 4, 2] -> [batch_size, 4]  
                labels = labels.view(-1, dataset_config.num_options).argmax(dim=1)  # [batch_size, 4] -> [batch_size, ]
                # ---------------------------------
                
                loss:torch.Tensor = criterion(logits, labels.long())

                if step % 300 == 0 and step!=0 and accelerator.is_main_process:  
                    # 打印训练过程中的预测分布
                    print_prediction_distribution(outputs,step,loss, dataset_config.num_options, logits, labels)
                total_loss += loss.detach().float().item() 
                
                accelerator.wait_for_everyone()
                
                # loss.backward()
                accelerator.backward(loss, retain_graph=True)
                
                
                should_update = accelerator.sync_gradients
                
                
                # 梯度累积  
                if should_update:  
                    # 在这里添加梯度裁剪，max_norm可以根据需要调整（常用值：1.0, 5.0, 10.0）  
                    accelerator.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.max_grad_norm,
                        norm_type=config.norm_type
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # 梯度累积
                # if (step) % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()
                    
                # 假设 step = 31, batch_size =2
                # (step+1)*2 = 64 = target batch_size
                
                
                
                
                # 确保在每次迭代后释放计算图  
                del outputs  
                del loss

                torch.cuda.empty_cache()  # 可选，如果内存占用过高
            
        global_step+=1
        accelerator.wait_for_everyone()
            
        # 在每个epoch结束后检查参数变化  
        if accelerator.is_main_process and epoch > 0:  
            check_param_after_epoch(model,param_monitor,epoch)
            
        if accelerator.is_main_process:
            print(f"begin epoch {epoch} evaluating...")    
            
        # eval_results = evaluate_baas_prompt(model, accelerator, eval_dataloader)
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
        # accelerator.log(
        #     {
        #         'epoch': epoch, 
        #         # 'avg_loss': avg_loss,
        #         **eval_results
        #     }
        # )

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
    
    # 判断模型名称
    
    # model_name = get_model_name_using_model(model)
    print("model name = ", model_name)

    # 保存权重
    # save_path = Config[SAVE_DIR][model_name][config.peft_method][dataset_name]
    # torch.save(model.state_dict(), save_path) 

    # wait every GPU processes to reach here
    torch.distributed.barrier()  

    # only the master process can save model
    # if torch.distributed.get_rank() == 0:  
    #     model.module.save_pretrained(save_path) 
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)  
    #     print(f"已创建新的权重存储路径: {save_path}") 
        
    # accelerator.save(model.state_dict(), save_path)   

        # tokenizer.save_pretrained('path_to_save_tokenizer')   


# def detect_param_grad_updates(model, epoch, step):
#     '''
#     检测可训练参数的梯度是否真的在更新
#     '''  
#     print(f"\n\n******************* Epoch:{epoch}, Step:{step}, Check Whether Gradient is Updating ***************8")
#     for name, param in model.named_parameters():  
#         if param.requires_grad:  
#             print(f"{name}'s parameter mean: {param.data.mean().item() if param.data is not None else 'None'}")  
#             print(f"{name}'s gradient norm: {param.grad.norm().item() if param.grad is not None else 'None'}") 
    
#     print("\n\n================== NaN Gradient Detection ========================================")
#     for name, param in model.named_parameters():  
#         if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):  
#             print(f"Gradient of {name} contains nan or inf at epoch:{epoch} step: {step}")
#     print("**************************************************************\n\n\n")
    

# def monitor_gradients(model, step):  
#     print(f"\n\n********************** Monitor Gradients for step={step}******************************8")
#     grad_stats = {}  
#     for name, param in model.named_parameters():  
#         if param.grad is not None:  
#             grad_norm = param.grad.norm().item()  
#             if grad_norm < 1e-8:  
#                 print(f"Warning: Very small gradient for {name}: {grad_norm}")  
#             grad_stats[name] = grad_norm  
    
#     # 检查梯度比例  
#     max_grad = max(grad_stats.values())  
#     min_grad = min(grad_stats.values())  
#     if max_grad / min_grad > 1000:  # 梯度比例阈值  
#         print(f"Warning: Large gradient ratio 'max_grad/min_grad' at step {step}: {max_grad/min_grad}")
    
#     print("*********************************************************************\n\n\n")


def evaluate_baas_prompt(
    model, 
    accelerator:Accelerator, 
    eval_dataloader:DataLoader=None
    )->Dict:
    
    model.eval()  
    all_preds = []  
    all_labels = []  
    
    for batch in eval_dataloader:  
        with torch.no_grad():  
        
            # 获取输入数据  
            # input_ids = batch['input_ids']  
            # attention_mask = batch['attention_mask']  
            # labels = batch['labels']  
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1) 
            labels = batch['labels']
        
            preds, labels = accelerator.gather_for_metrics(
                (preds, labels)
            )
    
            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  
    
    if accelerator.is_main_process:  
        metrics = {  
            'accuracy': accuracy_score(all_labels, all_preds),  
            'precision': precision_score(all_labels, all_preds, average='weighted'),  
            'recall': recall_score(all_labels, all_preds, average='weighted'),  
            'f1': f1_score(all_labels, all_preds, average='weighted')  
        }          
        # debug info
        print("\n****************** Evaluation Results:**************************")  
        print(f"Total samples evaluated: {len(all_labels)}")  
        print(f"Batch predictions distribution: {np.bincount(all_preds)}")  
        print(f"Batch labels distribution: {np.bincount(all_labels)}")   
        print("\n******************** Classification Report: ***********************")  
        print(classification_report(all_labels, all_preds))         
        print("*****************************************************************\n")
         
        
        return metrics 
    
    # 非主进程返回None  
    return None



if __name__ == "__main__":
    # model_name = "bert-large-uncased"
    # model_path = Config["models"]["bert-large-uncased"]["model_path"]

    
    # # 加载数据集
    # dataset_name = "commonsense_qa"
    
    args = parse_training_arguments()
    dataset_name =args.dataset_name
    model_name = args.model_name
    classes_initiate_method = args.classes_initiate_method
    model_path = get_model_path_by_model_name(model_name)

    dataset_path = Config["datasets"][dataset_name]
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification)
    
    max_seq_length = get_max_length_from_model(model)
    import math
    prefix_length = 10
    
    suffix_ratio = args.suffix_ratio
    suffix_ratio/=100
    suffix_length = math.floor(suffix_ratio * max_seq_length)
    
    hidden_size = get_hidden_size_using_model(model)

    config = BaasPromptConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        num_epochs=args.num_epochs,
        num_labels=2,
        all_layers=False,
        is_prefix=False,
        prefix_projection=True,
        prefix_hidden_size=hidden_size,
        encoder_hidden_size=hidden_size,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        batch_size=args.batch_size,
        debug=False,
        seq_cls_type="binary",
        classes_initiate_method = args.classes_initiate_method,
        train_size = args.train_size,
        mixed_precision=args.mixed_precision,
        suffix_ratio=args.suffix_ratio
        
        
    )
    
    args=ChainEncodingArguments(
        dataset=config.dataset_name,
        hidden_size=config.prefix_hidden_size, # 这个hidden_size最终会传给encode cot chain用到的 sentence transformer
        output_dir=f"./autocot/experiment/{dataset_name}",
        embedding_dir = f"./autocot/embeddings/{dataset_name}",
        context_dir = f"./autocot/context/{dataset_name}",
        temperature=0.7
    )
    train_baas_prompt(config, args)
    
    
    