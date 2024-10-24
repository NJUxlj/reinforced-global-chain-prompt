import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import evaluate
import os
from config import Config


from load import (
    preprocess_function_race_pt, 
    preprocess_function_race,
    load_dataset_from_huggingface,
    preprocess_race,
)


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

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support 




# 定义Prompt Encoder，用于生成可训练的提示向量  
class PromptEncoder(nn.Module):  
    def __init__(self, prompt_length, embedding_size, hidden_size):  
        super(PromptEncoder, self).__init__()  
        self.prompt_length = prompt_length  
        # 初始化可训练的提示embedding  
        self.prompt_embeddings = nn.Embedding(prompt_length, embedding_size)  
        # 将提示embedding映射到模型的hidden size  
        self.transform = nn.Sequential(  
            nn.Linear(embedding_size, hidden_size),  
            nn.Tanh(),  
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),  
        )  

    def forward(self, batch_size, device):  
        # 创建提示的索引  
        prompt_indices = torch.arange(self.prompt_length).long().to(device)  
        # 获取提示embedding  
        prompt_embeddings = self.prompt_embeddings(prompt_indices)  
        # 通过transform层  
        prompt_embeddings = self.transform(prompt_embeddings)  
        # 扩展batch维度  
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        return prompt_embeddings  

# 定义P-Tuning V2的BERT模型  
class PTuningV2BertForSequenceClassification(nn.Module):  
    def __init__(self, model, num_labels, prompt_length):  
        super(PTuningV2BertForSequenceClassification, self).__init__()  
        # 加载预训练的BERT模型  
        self.bert = model
        self.prompt_length = prompt_length  
        self.num_labels = num_labels  
        # 初始化Prompt Encoder  
        self.prompt_encoder = PromptEncoder(  
            prompt_length,   
            self.bert.config.hidden_size,   
            self.bert.config.hidden_size  
        )  
        # 冻结BERT模型的参数  
        for param in self.bert.parameters():  
            param.requires_grad = False  
        # 分类器  
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  

    def forward(self, input_ids, attention_mask, labels=None):  
        device = input_ids.device  
        batch_size = input_ids.size(0)  
        # 获取可训练的提示embedding  
        prompt_embeddings = self.prompt_encoder(batch_size, device)  
        # 获取原始的输入embedding  
        inputs_embeds = self.bert.embeddings(input_ids)  
        # 将提示embedding与输入embedding拼接  
        inputs_embeds = torch.cat([prompt_embeddings, inputs_embeds], dim=1)  
        # 调整attention mask  
        prompt_attention_mask = torch.ones(batch_size, self.prompt_length).to(device)  
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)  
        # 通过BERT模型  
        # outputs.shape = (batch_size, max_length, hidden_size)
        outputs = self.bert(  
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask  
        )  
        # 获取[CLS]位置的输出  
        # outputs.last_hidden_state.shape = (batch_size, max_length, hidden_size)
        pooled_output = outputs.last_hidden_state[:, 0]  
        # 通过分类器  
        logits = self.classifier(pooled_output)  
        loss = None  
        if labels is not None:  
            # 计算损失  
            loss_fn = nn.CrossEntropyLoss()  
            loss = loss_fn(logits, labels)  
        return {'loss': loss, 'logits': logits}  



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
        
        
def train_p_tuning_v2(model, tokenizer):
    

    # 初始化参数  
    model_name = 'bert-base-uncased'  
    num_labels = 4  # ['A', 'B', 'C', 'D']
    prompt_length = 30  
    batch_size = 2  
    num_epochs = 5  
    learning_rate = 5e-5  
    max_length = 512 - prompt_length
    
    
    # 加载数据集
    dataset_name = "race"
    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    # coarse-grained preprocessing
    ds, classes, tokenizer = preprocess_race(ds, tokenizer)

    processed_ds = ds.map(
        lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), # 从load.py导入  max_length = 492, 等下要加20个virtual tokens
        batched=True,
        num_proc=1,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",    
    )   
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]


    # # 创建数据集和数据加载器  
    # train_dataset = Racedataset(dataset['train'], tokenizer, prompt_length)  
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size)

    # 初始化模型  
    device = Config['device'] 
    # encapsulate the base model into a P-Tuning V2 model
    model = PTuningV2BertForSequenceClassification(model, num_labels, prompt_length)  
    model.to(device)  

    # 只优化Prompt Encoder和分类器的参数  
    optimizer = optim.Adam([  
        {'params': model.prompt_encoder.parameters()},  
        {'params': model.classifier.parameters()}  
    ], lr=learning_rate)  

    # 训练循环  
    model.train() 
    global_step = 0 
    for epoch in range(num_epochs):  
        total_loss = 0
        for batch in train_dataloader:  
            optimizer.zero_grad()  
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
            labels = batch['labels'].to(device)  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  
            loss = outputs['loss']  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item()
            
            # evaluate for each 5 batch-steps
            if global_step % 5 == 0:  
                model.eval()  
                all_preds = []  
                all_labels = []  
                with torch.no_grad():  
                    for val_batch in eval_dataloader:  
                        val_input_ids = val_batch['input_ids'].to(device)  
                        val_attention_mask = val_batch['attention_mask'].to(device)  
                        val_labels = val_batch['labels'].to(device)  
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
                model.train()  
            global_step+=1
              
        avg_loss = total_loss / len(train_dataloader)   
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")  

    # 保存模型
    save_path = Config['save_model_dir']['bert-base-uncased']['prompt-tuning-v2']['race']  
    torch.save({  
        'prompt_encoder_state_dict': model.prompt_encoder.state_dict(),  
        'classifier_state_dict': model.classifier.state_dict()  
    }, save_path)  






if __name__ == "__main__":
    '''
    
    '''
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    # 使用 AutoModel可以直接调用model.embedding. 否则不会暴露嵌入层
    model = AutoModel.from_pretrained(model_path, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model's current num_labels: {model.config.num_labels}") 
     
    model_config = model.config
    model_name_or_path = model_config.name_or_path
    print("model_name_or_path = ", model_name_or_path)
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    
    print("padding_side = ", padding_side)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)

    

    # 这里传model进去只是为了符合其他fine-tuning的写法，实际上这里不需要model
    train_p_tuning_v2(model, tokenizer)
    
    
    