import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os  
import shutil  
from config import Config
from dataclasses import dataclass
from collections import Counter
import time

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
    AutoModel,
    BertModel,
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)
from peft import (
    TaskType,
    PeftType,
    PeftConfig,
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

from accelerate import Accelerator

from swanlab.integration.accelerate import SwanLabTracker

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import (   
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score,  
    classification_report,
    confusion_matrix,
)  

import evaluate
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


@dataclass  
class PromptTuningTrainerConfig:  
    """MCQA任务的P-tuning V2配置"""  
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "prompt-tuning"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 100                        # 前缀长度  
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 2
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                     # 最大序列长度  
    learning_rate: float = 0.3   # 0.3                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768                 # MLP中的P_theta'  即，MLP输入的隐单元维度  huggingface 默认它==encoder_hidden_size
    encoder_hidden_size:int = 768  # 编码器(bert)的隐藏层大小
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减  
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # beta2: 二阶矩估计的指数衰减率（默认0.999）
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = Adam
    
    seed:int=42
    debug:bool=False
    train_size:int=22000
    mixed_precision:bool = False




def train_prompt_tuning(config:PromptTuningTrainerConfig):
    
    fix_seed(config.seed)
    setup_distributed()
    setup_cuda_debug_environment()
    print("\n\n",config,"\n\n")
    print_training_info(config)
    # 初始化参数  
    model_name = config.model_name
    
    # 配置num_labels
    model, tokenizer = prepare_model_tokenizer(config.model_path, AutoModelForSequenceClassification, config.model_path, num_labels=config.num_labels )
    
    num_labels = config.num_labels
    
    prefix_length = config.prefix_length
    batch_size = config.batch_size
    lr = config.learning_rate
    num_epochs = config.num_epochs
    
    max_length = config.max_seq_length
    print(f"before inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    max_length = max_length - prefix_length
    print(f"After inserting prompt tokens, {model_name}'s max length = {max_length}")
    
    # if model.config.model_type=='roberta':
    #     max_length-=3
    
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
    processed_ds = preprocess_dataset_peft(dataset_name, config.model_path, max_length=max_length, train_size=config.train_size)
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]  
    
    # train_ds = train_ds[:5000]
    
    
    print("training set size = ", len(train_ds))
    print("eval set size = ", len(eval_ds))
    
    # 使用DistributedSampler进行数据分布  
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
            sampler = eval_sampler
        )
    
    
    
    
    # Prompt-tuning
    peft_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=prefix_length, 
        token_dim = config.prefix_hidden_size,  
        # num_transformer_submodules=1,   # 此参数在最新版本中已不再使用
        # In many cases, this is set to 1, 
        # meaning that the prompt tuning will interact with a single submodule, 
        # often the self-attention submodule, to inject the prompt information into the model.

        # num_attention_heads=12,   # 自动根据模型类型指定
        # num_layers=1,    # 自动指定
        prompt_tuning_init = "TEXT",  # 使用文本初始化prompt 
        prompt_tuning_init_text = "Classify the answer of this question among  A, B, C, and D",
        tokenizer_name_or_path = config.model_path,  # 路径 or 模型名称 or 模型本地路径
    )
    
    # Input Shape: (batch_size, total_virtual_tokens)

    # Output Shape: (batch_size, total_virtual_tokens, token_dim)
    
    model = get_peft_model(model, peft_config)
    model.to(accelerator.device)
    model.print_trainable_parameters()
    
    # for param in model.base_model.parameters():  
    #     param.requires_grad = False
    

    
    if accelerator.is_main_process:
        print("===================== Weight Summary =========================")
        for name, param in model.named_parameters():  
            if 'prompt_encoder' in name or 'classifier' in name:  
                # param.requires_grad = True 
                print("weight name = {}, shape = {}, requires_grad = {}".format(name, param.shape, param.requires_grad))
            else:
                # param.requires_grad = False
                pass
        
        print("=============================================================")
        
        
    if accelerator.is_main_process:
        # 打印可训练参数  
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        print(f"Trainable parameters (self-calculated): {trainable_params}")  


    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
        
        
    device = accelerator.device
    global_step = 0
    best_accuracy = 0 
    
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
            # with accelerator.accumulate(model):  # 使用梯度累积 
                    
                # 定期清理缓存
                # torch.cuda.empty_cache()
                     
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            labels = batch["labels"]  
            
            
            # print("batch.keys = \n",batch.keys())
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            if config.debug:
                print("max_sequence_length = ", config.max_seq_length)
                print("model.config.max_position_embeddings = ", model.module.config.max_position_embeddings)
                print("batch[\"input_ids\"].shape = ",batch["input_ids"].shape)
                print("batch[\"attention_mask\"].shape = ",batch["attention_mask"].shape)
            # time.sleep(10000)
            
            
            
            bz, seq_length = batch["attention_mask"].shape
            # position_ids = torch.arange(  
            #     2, seq_length+2,   
            #     dtype=torch.long,   
            #     device=accelerator.device 
            # ).unsqueeze(0).expand(bz, -1)   
            
            # position_ids = create_position_ids_safe(batch.get("attention_mask"), config.max_seq_length, padding_idx=1)
            # print("position_ids.shape = ",position_ids.shape)
            # print("position_ids.value = \n", position_ids)
            # # 1. 检查position_ids的范围  
            # print("Position IDs max:", position_ids.max())  
            # print("Position IDs min:", position_ids.min())  
            
            # time.sleep(10000)
            
            
            # batch['position_ids'] = position_ids
            
            if config.debug:
                debug_cuda_sync("start model forward")
            model: RobertaForSequenceClassification
            
            outputs = model.forward(**batch)
            # 不需要根据样本数量设置权重（因为1:3是必然的）  
            # 但可以设置稍微高一点的正样本权重来增强学习信号  
            # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
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
                
            # loss= outputs.loss
            total_loss += loss.detach().float().item() 

            accelerator.wait_for_everyone()
            
            accelerator.backward(loss, retain_graph=True)
            
            should_update = accelerator.sync_gradients
            
            if should_update:  
                # 在这里添加梯度裁剪，max_norm可以根据需要调整（常用值：1.0, 5.0, 10.0）  
                # accelerator.clip_grad_norm_(
                #     model.parameters(), 
                #     max_norm=config.max_grad_norm,
                #     norm_type=config.norm_type
                #     )
                if step % 1000 == 0 and step!=0 and accelerator.is_main_process:
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
        eval_results = evaluate_prompt_tuning(model, eval_dataloader, accelerator, dataset_config)
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
        
            # if step % 100 ==0 and step !=0:
            #     if accelerator.is_local_main_process: 
            #          # 检查模型参数变化  
            #         for name, param in model.named_parameters():  
            #             if param.requires_grad:  
            #                 print(f"{name}: mean={param.data.mean().item():.4f}, "  
            #                     f"std={param.data.std().item():.4f}, "  
            #                     f"grad_mean={param.grad.mean().item() if param.grad is not None else 'None'}") 
                    

    #         if step == len(train_dataloader)-1:
    #             pass  
    #             # results, model, accelerator = evaluate_prompt_tuning(model, eval_dataloader, accelerator)  
                
    #             # model=model
    #             # accelerator = accelerator
                
    #             # if accelerator.is_main_process:             
    #             #     logger.info({
    #             #         'epoch': epoch, 
    #             #         'loss': loss.item(), 
    #             #         'accuracy':results["accuracy"], 
    #             #         "precision": results["precision"], 
    #             #         "recall": results["recall"], 
    #             #         "f1": results["f1"], 
    #             #     })
        
    #     # 每轮完全结束以后进行评价
    #     results = evaluate_prompt_tuning(model, eval_dataloader, accelerator, device)  
        
    #     if accelerator.is_main_process:             
    #         logger.info({
    #             'epoch': epoch, 
    #             'loss': loss.item(), 
    #             'accuracy':results["accuracy"], 
    #             "precision": results["precision"], 
    #             "recall": results["recall"], 
    #             "f1": results["f1"], 
    #         }) 
            
    #     if accelerator.is_local_main_process: 
    #         avg_loss = total_loss / len(train_dataloader)   
    #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  
    
    # # 所有epoch结束以后进行总评
    # results = evaluate_prompt_tuning(model, eval_dataloader, accelerator)
    # if accelerator.is_main_process:
    #     logger.info({
    #                 'accuracy':results["accuracy"], 
    #                 "precision": results["precision"], 
    #                 "recall": results["recall"], 
    #                 "f1": results["f1"], 
    #             }) 


    # 保存权重
    print("model name = ", model_name)
    # save_path = Config[SAVE_DIR][model_name][config.peft_method][dataset_name]
    
    # wait every GPU processes to reach here
    # torch.distributed.barrier()  
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)  
    #     print(f"已创建新的权重存储路径: {save_path}") 
    
    # accelerator.save({  
    #     'prefix_encoder': model.prefix_encoder.state_dict(),  
    #     'classifier': model.classifier.state_dict()  
    # }, save_path) 
    
    
    
    # # 只让主进程处理文件操作  
    # if accelerator.is_main_process:  
    #     # 如果目录已存在，删除整个目录及其内容  
    #     if os.path.exists(save_path):  
    #         try:  
    #             shutil.rmtree(save_path)  
    #             print(f"已删除旧的权重目录: {save_path}")  
    #         except Exception as e:  
    #             print(f"删除旧权重目录时出错: {e}")  
        
    #     # 创建新的目录  
    #     try:  
    #         os.makedirs(save_path)  
    #         print(f"已创建新的权重存储路径: {save_path}")  
    #     except Exception as e:  
    #         print(f"创建新目录时出错: {e}")   
        
    
    # model_state_dict = get_peft_model_state_dict(model.module)
    # accelerator.save(model_state_dict, save_path)  




def evaluate_prompt_tuning(
        model, 
        eval_dataloader, 
        accelerator:Accelerator, 
        dataset_config:DatasetConfig,
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
    
    # 添加更多指标统计  
    total_questions = 0  
    correct_questions = 0  
    all_probs = []  # 保存原始概率分布  
    
    for batch in eval_dataloader:  
        with torch.no_grad():  
            outputs = model(**batch)
            # --------------------
            logits = outputs.logits # [batch_size x 4, 2] 
            probs = torch.softmax(logits, dim=1)[:, 1]  # 获取正类概率   # shape = (batch_size * 4, 1)
            # 每4个样本为一组重塑  
            probs = probs.view((-1, dataset_config.num_options))  # shape = (batch_size, 4)  每个问题对应4个选项，每个选项都有一个预测为1的概率 [0.2, 0.1, 0.5, 0.2]， 此时我们认为选项3是正确的
            labels = batch['labels'].view((-1, dataset_config.num_options))  # shape = (batch_size, 4)
            
            all_probs.append(probs.cpu())  # 保存原始概率分布
            
            # 检查每道题是否预测正确  
            pred_answers = probs.argmax(dim=1)  # 每道题预测的答案  比如 选项3 对应了 label = 2  shape = (batch_size, 1)
            true_answers = labels.argmax(dim=1)  # 每道题的正确答案  比如[0, 0 ,1, 0] -> argmax -> 2  shape = (batch_size, 1)
            preds = pred_answers
            labels = true_answers
            
            # 统计正确题目数  
            total_questions += batch_size  
            correct_questions += (pred_answers == true_answers).sum().item()  
            # --------------------------
            # preds = outputs.logits.argmax(dim=-1) 
            # labels = batch['labels']
            
            preds, labels = accelerator.gather_for_metrics(
                (preds, labels)
            )
            
        
            all_preds.append(preds.cpu())  
            all_labels.append(labels.cpu())   
            
    
    all_preds = torch.cat(all_preds).numpy()  
    all_labels = torch.cat(all_labels).numpy() 
    all_probs = torch.cat(all_probs).numpy()  # shape = (batch_size, 4)
            
    
    
    # 计算评价指标  
    metrics = {  
        'accuracy': accuracy_score(all_labels, all_preds),  
        'precision': precision_score(all_labels, all_preds, average='weighted'),  
        'recall': recall_score(all_labels, all_preds, average='weighted'),  
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        "question accuracy": correct_questions/total_questions,
        "mean_confidence": np.mean(all_probs.max(axis=1)), # 平均预测置信度
    }     
    
    accelerator.wait_for_everyone()     
    
    if accelerator.is_main_process: 
        
        # debug info
        print("\n****************** Evaluation Results:**************************")  
        print(f"Total questions evaluated: {total_questions}")  
        print(f"Correct questions: {correct_questions}")  
        print(f"Question-level accuracy: {correct_questions/total_questions:.4f}") 
        print(f"Total questions evaluated: {len(all_labels)}")  
        # print(f"Batch predictions distribution: {np.bincount(all_preds)}")  
        # print(f"Batch labels distribution: {np.bincount(all_labels)}")       
        print("*****************************************************************\n")
        
        # 预测分布分析  
        print("\nPrediction Distribution:")  
        for i in range(dataset_config.num_options):  
            count = (all_preds == i).sum()  
            print(f"Option {i}: {count} ({count/len(all_preds):.2%})")  
        
        # 混淆矩阵  
        print("\nConfusion Matrix:")  
        print(confusion_matrix(all_labels, all_preds))  
        
        print("\n******************** Classification Report: ***********************")  
        print(classification_report(all_labels, all_preds))    
        
    
    # 恢复模型原始状态  
    model.train(training_state) 
        
    return metrics 
        
    



def evaluate_legacy(model, accelerator, train_dataloader):
    '''
    very old model evaluation code, leave it here for future usage
    '''
    #     all_preds = []  
    #     all_labels = []  
    #     with torch.no_grad():  
    #         for val_batch in eval_dataloader:  
    #             val_input_ids = val_batch['input_ids']
    #             val_attention_mask = val_batch['attention_mask']
    #             val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)  

    #             logits = val_outputs.logits  # shape = (batch_size, num_labels)
    #             val_labels = val_batch['labels']
                
    #             if accelerator.use_distributed:  
    #                 accelerator.wait_for_everyone()
    #                 logits = accelerator.gather_for_metrics(logits)
    #                 val_labels = accelerator.gather_for_metrics(val_labels)
                
    #             preds = torch.argmax(logits, dim=1).cpu().numpy()  
    #             labels_cpu = val_labels.cpu().numpy()  
                        
    #             all_preds.extend(preds)  
    #             all_labels.extend(labels_cpu)  
    #     # 计算评价指标  
    #     accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
    #     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')  
    #     # print(f"Step {global_step}, Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")  

        # if accelerator.is_main_process:             
            # logger.info({
            #     'epoch': epoch, 
            #     'loss': loss.item(), 
            #     'accuracy':results["accuracy"], 
            #     "precision": results["precision"], 
            #     "recall": results["recall"], 
            #     "f1": results["f1"], 
            # }) 
            
            # logger.info({
            #     'epoch': epoch, 
            #     'loss': loss.item(), 
            #     'accuracy':accuracy, 
            #     "precision": precision, 
            #     "recall": recall, 
            #     "f1": f1, 
            # })  
            # print()
            # 保存最佳模型  
            # if results['f1'] > best_accuracy:  
            #     best_accuracy = results['f1']  
            #     accelerator.wait_for_everyone()  
            #     unwrapped_model = accelerator.unwrap_model(model)  
                
            #     save_path = Config[SAVE_DIR][model_name][config.peft_method][dataset_name]
                
            #     if not os.path.exists(os.path.dirname(save_path)):  
            #         os.makedirs(os.path.dirname(save_path))  
                
            #     # 只保存prompt embedding  
            #     prompt_state_dict = {}  
            #     for name, param in unwrapped_model.named_parameters():  
            #         if 'prompt' in name:  
            #             prompt_state_dict[name] = param.data.cpu()
                        
            #     unwrapped_model.save_pretrained(  
            #         save_path,  
            #         state_dict=prompt_state_dict,  # 只保存prompt相关的权重  
            #     )  
            
        # accelerator.wait_for_everyone() 
        # model.train()  
    
    if accelerator.is_local_main_process: 
        # progress_bar.close()  
        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 加载保存的prompt embedding的函数  
def load_trained_prompt(base_model, prompt_path):  
    """  
    加载训练好的prompt embedding  
    Args:  
        base_model: 原始基座模型  
        prompt_path: 保存的prompt权重路径  
    Returns:  
        加载了训练好的prompt embedding的模型  
    """  
    # 加载PEFT配置  
    config = PeftConfig.from_pretrained(prompt_path)  
    
    # 创建PEFT模型  
    model = get_peft_model(base_model, config)  
    
    # 加载训练好的prompt权重  
    state_dict = torch.load(os.path.join(prompt_path, 'adapter_model.bin'))  
    model.load_state_dict(state_dict, strict=False)  
    
    return model  


if __name__ == '__main__':
    '''

    '''
    # model_path = Config["models"]["bert-large-uncased"]["model_path"]
    # model_name = "bert-large-uncased"
    
    # model_path = Config['models']['qwen']['Qwen2.5-1.5B']["model_path"]
    # model_name = 'Qwen2.5-1.5B'
    
    # dataset_name = "race"
    args = parse_training_arguments()
    dataset_name =args.dataset_name
    model_name = args.model_name
    model_path = get_model_path_by_model_name(model_name)
    train_size = args.train_size
    batch_size = args.batch_size
    mixed_precision = args.mixed_precision
    
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification, model_path, num_labels=2)

    max_seq_length = get_max_length_from_model(model)
    hidden_size = get_hidden_size_using_model(model)

    config = PromptTuningTrainerConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name= dataset_name,
        max_seq_length=max_seq_length,
        num_epochs=5,
        num_labels=2,
        prefix_hidden_size=hidden_size,
        encoder_hidden_size=hidden_size,
        batch_size=batch_size,
        debug=False,
        train_size=train_size
    )


    train_prompt_tuning(config)
    