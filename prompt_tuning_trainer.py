import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os  
import shutil  
from config import Config
from dataclasses import dataclass
from collections import Counter

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
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_utils import EvalPrediction 
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    classification_report  
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
    
    
    
@dataclass  
class DataCollatorForPromptTuning:  
    """  
    数据整理器，处理批量数据  
    """  
    tokenizer: PreTrainedTokenizerBase  
    padding: Union[bool, str, PaddingStrategy] = True  
    max_length: int = None  
    pad_to_multiple_of: int = None  

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:  
        # 提取输入特征  
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]  
        attention_mask = [{"attention_mask": feature["attention_mask"]} for feature in features]  
        
        # 批量填充  
        batch = self.tokenizer.pad(  
            input_ids,  
            padding=self.padding,  
            max_length=self.max_length,  
            pad_to_multiple_of=self.pad_to_multiple_of,  
            return_tensors="pt",  
        )  
        
        # 添加标签  
        if "label" in features[0].keys():  
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)  
            
        return batch  

class PromptTuningTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")  
            outputs = model(**inputs)  
            logits = outputs.logits  
            loss_fct = nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        
        
def compute_metrics(eval_pred: EvalPrediction):  
    """  
    计算评估指标  
    """  
    metric = evaluate.load("accuracy")  
    precision_metric = evaluate.load("precision")  
    recall_metric = evaluate.load("recall") 
    f1_metric = evaluate.load("f1")  
    
    
    predictions = eval_pred.predictions  
    labels = eval_pred.label_ids  
    
    # 获取预测类别  
    preds = np.argmax(predictions, axis=1)  
    
    results = {}  
    # 计算准确率  
    results.update(metric.compute(predictions=preds, references=labels)) 
    
    # 计算精确率（使用weighted average处理类别不平衡）  
    results.update(precision_metric.compute(predictions=preds, references=labels, average="weighted"))  
    
    # 计算召回率  
    results.update(recall_metric.compute(predictions=preds, references=labels, average="weighted"))  
    # 计算F1分数  
    results.update(f1_metric.compute(predictions=preds, references=labels, average="weighted"))  
    
    # 打印详细结果  
    print("\nEvaluation Results:")  
    for metric_name, value in results.items():  
        print(f"{metric_name}: {value:.4f}")  
    
    return results  



def setup_trainer(config: PromptTuningTrainerConfig, model, train_dataset, eval_dataset, tokenizer):  
    """  
    设置和配置Trainer  
    """  
    
    # 训练参数配置  
    training_args = TrainingArguments(  
        output_dir=f"prompt_tuning_{config.model_name}",  
        learning_rate=config.learning_rate,  # 相对较大的学习率  
        per_device_train_batch_size=config.batch_size,  
        per_device_eval_batch_size=config.batch_size,  
        num_train_epochs=config.num_epochs,  
        weight_decay=config.weight_decay,  
        evaluation_strategy="epoch",  
        save_strategy="epoch",  
        # load_best_model_at_end=True,  
        # metric_for_best_model="f1",  
        # greater_is_better=True,  
        push_to_hub=False,  
        remove_unused_columns=False,  
        gradient_accumulation_steps=4,  # 梯度累积  
        warmup_ratio=0.1,  
        logging_strategy="epoch",
        logging_dir=Config['logging_dir'][model_name][config.peft_method][dataset_name],
        # logging_steps=100,  
        # report_to=["tensorboard"],  
        # save_total_limit=2,  # 只保存最好的2个检查点  
    )  
    
    # 数据整理器  
    data_collator = DataCollatorForPromptTuning(  
        tokenizer=tokenizer,  
        padding=True,  
        max_length=config.max_seq_length  
    )  
    
    # 初始化Trainer  
    trainer = PromptTuningTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,  
        # data_collator=data_collator,  
        tokenizer=tokenizer,  
        compute_metrics=compute_metrics,    
    )  
    
    return trainer  


def train_prompt_tuning(config:PromptTuningTrainerConfig):
    
    # fix_seed(config.seed)t_t
    # setup_distributed()
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
    
    
    # 加载数据集
    dataset_name = config.dataset_name
    dataset_path = get_dataset_path_by_name(dataset_name)
    
    processed_ds = preprocess_dataset_peft(dataset_name, max_length=max_length)
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]  
    
    
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
    model.print_trainable_parameters()
    
    # for param in model.base_model.parameters():  
    #     param.requires_grad = False
    


    # optimizer = torch.optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=lr
    # )
    
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.warmup_steps,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )



    trainer = setup_trainer(  
        config,  
        model,  
        train_ds,  
        eval_ds,  
        tokenizer  
    )  
    
    # 开始训练  
    print("Starting training...")  
    train_results = trainer.train()  
    
    # 保存最终模型  
    trainer.save_model(f"final_model_{config.model_name}")  
    
    # 打印训练结果  
    print("Training results:")  
    print(train_results)  
    
    # 进行最终评估  
    eval_results = trainer.evaluate()  
    print("Final evaluation results:")  
    print(eval_results)  
    
    return trainer, eval_results  
    
    



def evaluate_prompt_tuning(
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
    model_path = Config["models"]["bert-large-uncased"]["model_path"]
    model_name = "bert-large-uncased"
    
    # model_path = Config['models']['qwen']['Qwen2.5-1.5B']["model_path"]
    # model_name = 'Qwen2.5-1.5B'
    
    dataset_name = "sciq"

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
    )


    trainer, eval_results = train_prompt_tuning(config)

    print("eval results = \n", eval_results)
    