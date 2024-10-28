from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    AutoTokenizer,  
    AutoModelForSequenceClassification,  
    Trainer,  
    TrainingArguments  
)
# from transformers import PromptForSequenceClassification  
from peft import (
    PromptTuningInit, 
    PromptTuningConfig,
    get_peft_model,
    TaskType
)

from load import (
    preprocess_function_race_pt, 
    preprocess_function_race,
    load_dataset_from_huggingface,
    preprocess_race,
)

from training_hub import (
    prepare_model_tokenizer, 
)

from datasets import (
    # Dataset,
    load_dataset,
)
from torch.utils.data import (
    DataLoader,
    Dataset
)

from config import Config

import torch
import torch.nn as nn
import numpy as np
import csv
import evaluate
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Dict




device = Config['device']

# # 初始化模型  
# model_path = Config["models"]["bert-base-uncased"]["model_path"]
# model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()  
# tokenizer = AutoTokenizer.from_pretrained(model_path)





# 3. 修改模型的输入嵌入函数，添加前缀和后缀Prompt Tokens  
class BidirectionalPromptModel(torch.nn.Module):  
    def __init__(self, model, prefix_embeddings, suffix_embeddings, num_prefix_tokens, num_suffix_tokens):  
        super(BidirectionalPromptModel, self).__init__()  
        self.model = model  
        self.prefix_embeddings = prefix_embeddings  
        self.suffix_embeddings = suffix_embeddings 
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.embedding_layer = self.model.get_input_embeddings()  
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):  
        # 原始输入嵌入
        # print(f"input_ids.shape = {input_ids.shape}")
        input_ids = input_ids.squeeze(1) 
        inputs_embeds = self.embedding_layer(input_ids)  
        
        batch_size = inputs_embeds.size(0)  
        
        # 将前缀和后缀Prompt Embeddings扩展到batch维度  
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        suffix_embeds = self.suffix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        
        # print(f"prefix.shape = {prefix_embeds.shape}")
        # print(f"suffix.shape = {suffix_embeds.shape}")
        # print(f"inputs_embeds.shape = {inputs_embeds.shape}")

        # 拼接前缀、原始输入和后缀嵌入  
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds, suffix_embeds], dim=1)  # (4, 522, 768)
        
        # 调整attention_mask  
        if attention_mask is not None:  
            prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device)  
            suffix_mask = torch.ones(batch_size, self.num_suffix_tokens, device=device)  

            # print(f"attention_mask.shape = {attention_mask.shape}")
            # print(f"prefix_mask.shape = {prefix_mask.shape}")
            # print(f"suffix_mask.shape = {suffix_mask.shape}")
            
            attention_mask = attention_mask.squeeze(1)
            attention_mask = torch.cat([prefix_mask, attention_mask, suffix_mask], dim=1)  # (4, 522)
            
            # print(f"attention_mask.shape after concat = {attention_mask.shape}") 
        
        if token_type_ids is not None:  
            prefix_type_ids = torch.zeros(batch_size, self.num_prefix_tokens, device=device)
            suffix_type_ids = torch.zeros(batch_size, self.num_suffix_tokens, device=device)
            
            # print(f"token_type_ids.shape = {token_type_ids.shape}")
            # print(f"prefix_type_ids.shape = {prefix_type_ids.shape}")
            # print(f"suffix_type_ids.shape = {suffix_type_ids.shape}")
            
            token_type_ids = token_type_ids.squeeze(1)
            token_type_ids = torch.cat([prefix_type_ids, token_type_ids, suffix_type_ids], dim=1)
            
            token_type_ids = token_type_ids.long() # (4, 522)
            
            # print(f"token_type_ids.shape after concat = {token_type_ids.shape}")

        
        # 调用原始模型的forward方法  
        outputs = self.model(  
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids,
            labels=labels  
        )  
        
        return outputs  

    def print_trainable_parameters(self):  
        """print trainable parameters' number and ratio"""  
        trainable_params = 0  
        all_params = 0  
        for name, param in self.named_parameters():  
            num_params = param.numel()  
            all_params += num_params  
            if param.requires_grad:  
                trainable_params += num_params  
        print(f"trainable param number: {trainable_params}")  
        print(f"total param number: {all_params}")  
        print(f"trainable param ratio: {100 * trainable_params / all_params:.2f}%")  





def initialize_suffix_prompts(ds, tokenizer, num_suffix_tokens, embedding_size):  
    """  
     use Auto-CoT reasoning steps as the suffix prompts
    """  
    # 假设我们已经有AutoCoT生成的推理步骤steps，并将其保存为文本列表  
    steps_list: List[str] = []  
    
    for sample in tqdm(ds['train']):
        # generate AutoCoT-style input  
        text = f"Article: {sample['article']}\n\nQuestion: {sample['question']}\n\nOptions: {', '.join(sample['options'])}\n\nAnswer:" 
        
        # use pretrained model to generate reasoning steps as a string
        steps = "Generated reasoning steps for the sample."  
    
    
    suffix_prompt_embeddings = torch.tensor(cluster_centers, requires_grad=True).cuda()  
    return suffix_prompt_embeddings
    



def initialize_prefix_prompts(ds, tokenizer, num_prefix_tokens, embedding_size, classes, K=5): 
    """
    use article classification tokens' weighted sum as prefix prompts
    """
    
    # import the article classification model
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    num_classes = 10
    device = Config['device']
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes).to(device)  

    
    # get all articles
    articles:List[str] = ds['train']['article']   # shape = (n,)
    encoded_articles:Dict[str, torch.Tensor] = tokenizer(articles, padding=True, truncation=True, return_tensors = 'pt') 
    input_ids:torch.Tensor = encoded_articles['input_ids'].to(device) # shape = (n, max_length)
    attention_mask = encoded_articles['attention_mask'].to(device)
        
    # get classification scores
    with torch.no_grad():
        # outputs = {"logits", "hidden_states", "attentions"}
        # logits：形状为 (n, 10) 的张量，表示每个输入样本在每个类别的 logits。
        # hidden_states (if open, it represents the hidden states of all layers)：一个包含多个张量的元组，每个张量的形状为 (n, max_length, hidden_size)。
        # attentions (if open, it represents the attention weights of all layers)：一个包含多个张量的元组，每个张量的形状为 (n, num_heads, max_length, max_length)。
        outputs:Dict = classification_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # map the predictions to the textual labels
    labels:List[str] = [classes[[pred]] for pred in predictions]
    
    # transfer textual labels to embeddings
    encoded_labels = tokenizer(labels, padding="max_length", truncation=True, max_length=1, return_tensors = 'pt')
    labels_input_ids = encoded_labels['input_ids'].to(device) # shape = (n, 1)
    
    print("labels_input_ids = ", labels_input_ids)
    
    # get the embeddings of all examples' labels
    embeddings = classification_model.get_input_embeddings()(labels_input_ids).mean(dim=1)  # (n, embedding_size)  
    
    # get the pooled embedding of all the class label embeddings
    pooled_embedding = embeddings.mean(dim=0) # shape = (embedding_size)
    
    prefix_embeddings = torch.zeros(num_prefix_tokens, embedding_size, device=device)
    
    for i in range(num_prefix_tokens):  
        prefix_embeddings[i] = pooled_embedding
    
    prefix_prompt_embeddings = torch.nn.Parameter(prefix_embeddings, requires_grad=True)   # (num_prefix_tokens, embedding_size)

    return prefix_prompt_embeddings



def get_classes_for_dataset(dataset_path, K=5):
    '''
    get the label collection of some specific dataset
        
    Args:
        dataset_path: the path of dataset, should be in [race, race-m, race-h, multirc, arc]
        K: number of classes in the dataset
    
    Procedure:
        1. load the dataset
        2. load the first 1000 examples from the training set.
        3. get the sequence embedding of each example.
        4. get the avearage pooling of the 1000 sequence embeddings, for every 1000 examples, so there are in total len(train_data)//1000 pooled embeddings.
        5. calculate the cosine similarity between the pooled embedding and every token in the vocab, average the similarity on len(train_data)//1000 pooled embeddings.
        6. for each token in the vocab, we multiply the similarity by the frequency of the token in the training set. (TF-AS)
        6. select the top-K tokens in the vocab as the class labels, with the highest TF-AS score.
    
    Return:
        classes: a list of str, which contains the class labels
    '''
    ds = None
    classes = None
    
    def reformat_input(dataset_path, ds):
        if dataset_path == Config["datasets"]["race"]:
            
        elif dataset_path == Config["datasets"]["race-m"]:
            
        elif dataset_path == Config["datasets"]["race-h"]:
            
        elif dataset_path == Config["datasets"]["multirc"]:
        elif dataset_path == Config["datasets"]["arc"]:
        elif dataset_path == Config["datasets"]["dream"]:
        else:
            raise ValueError("dataset_path not supported, we can not reformat dataset using a wrong name, please change another in [race, race-m, race-h, multirc, arc]")
        
    
    def extract_labels(ds):
        '''
        get the label collection of the dataset
        ''' 
        
    
    if dataset_path == Config["datasets"]["race"]:
        ds = load_dataset_from_huggingface(dataset_path, "all")
        
    elif dataset_path == Config["datasets"]["race-m"]:
        ds = load_dataset_from_huggingface(dataset_path, "middle")
        
    elif dataset_path == Config["datasets"]["race-h"]:
        ds = load_dataset_from_huggingface(dataset_path, "high")
        
    elif dataset_path == Config["datasets"]["multirc"]:
        ds = load_dataset_from_huggingface(dataset_path)
    elif dataset_path == Config["datasets"]["arc"]:
        ds = load_dataset_from_huggingface(dataset_path)
    elif dataset_path == Config["datasets"]["dream"]:
        ds = load_dataset_from_huggingface(dataset_path)
    else:
        raise ValueError("dataset_path not supported, plase change another in [race, race-m, race-h, multirc, arc]")
    
    return classes


def get_label_collection_for_class(dataset_path, classes:List[str]):
    '''
    将原标签列表中的每个标签扩展成一个标签集合
    1. 训练一个线性层，将经过MLM的mask token分类到原有的标签
    
    return dict["class1" : set(label1, label2, ...)]
    '''

def train_bidirectional_prompt_tuning(model, tokenizer):
    device = Config['device']

    K=5
    # 定义双向Prompt Tuning的参数       
    num_prefix_tokens = 5   # 前缀Prompt Tokens的数量  
    num_suffix_tokens = 5   # 后缀Prompt Tokens的数量  
    embedding_size = model.get_input_embeddings().embedding_dim

    batch_size = Config['batch_size']
    lr = 3e-2
    num_epochs = Config['num_epochs']
    max_length = 512 - num_prefix_tokens - num_suffix_tokens


    # 1. initialize the trainable prefix prompt embeddings
    prefix_prompt_embeddings = torch.nn.Parameter(
        torch.rand(num_prefix_tokens, embedding_size,requires_grad=True, device= device),   # (num_prefix_tokens, embedding_size)
    )

    # 2. initialize the trainable suffix prompt embeddings
    suffix_prompt_embeddings  = torch.nn.Parameter(  
        torch.rand(num_suffix_tokens, embedding_size,requires_grad=True, device= device),   # (num_suffix_tokens, embedding_size)
    )

    # 4. 创建带有双向Prompt的模型实例  
    bidirectional_prompt_model = BidirectionalPromptModel(  
        model=model,  
        prefix_embeddings=prefix_prompt_embeddings,  
        suffix_embeddings=suffix_prompt_embeddings,
        num_prefix_tokens=num_prefix_tokens, 
        num_suffix_tokens=num_suffix_tokens,
    ).to(device)  

    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    
    # coarse-grained preprocessing
    ds, classes, tokenizer = preprocess_race(ds, tokenizer)
    
    # fine-grained preprocessing
    # the preprocessed dataset only contains ["input_ids", "attention_mask", "labels"]
    processed_ds = ds.map(
        lambda examples: preprocess_function_race(examples, max_length=max_length, tokenizer=tokenizer), # 从load.py导入
        batched=True,
        num_proc=1,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    
    train_ds = processed_ds["train"]
    eval_ds = processed_ds["test"]
    

    print("dataset is preprocessed successfully ~~~")
    
    
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size)


    
    # make sure to frozen the base model parameters
    for param in bidirectional_prompt_model.model.parameters():  
        param.requires_grad = False  
        
    # make sure that the prefix and suffix tokens is trainable
    bidirectional_prompt_model.prefix_embeddings.requires_grad = True  
    bidirectional_prompt_model.suffix_embeddings.requires_grad = True 
    
    
    bidirectional_prompt_model.print_trainable_parameters()
    
    # for name, param in bidirectional_prompt_model.named_parameters():  
    #   print(f"{name}: requires_grad = {param.requires_grad}") 
    # print("============================")
    # for param in bidirectional_prompt_model.parameters():  
    #   print(f"param = {param}: requires_grad = {param.requires_grad}")

    # make sure that the fine-tuning will only update virual tokens
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, bidirectional_prompt_model.parameters()), 
        lr=lr
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model = bidirectional_prompt_model 
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print(f"Batch labels: {batch['labels']}") 
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {"input_ids": tensor([[101, 7592, 2199, 2, ...], [101, 7592, 2199, ...]]), "attention_mask": tensor([[1, 1, 1,  ..., 0, 0, 0], [1, 1, 1, ...]])}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # evaluate for each 5 batch-steps
            if step == len(train_dataloader)-1:  
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
            


    # 保存权重
    save_path = Config['save_model_dir']['bert-base-uncased']['bidirectional-prompt-tuning']['race']
    # torch.save(model.state_dict(), save_path) 


    model.save_pretrained(save_path)
    # tokenizer.save_pretrained('path_to_save_tokenizer')   









if __name__ == "__main__":
    '''
    
    '''
    model_path = Config["models"]["bert-base-uncased"]["model_path"]
    # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(f"Model's current num_labels: {model.config.num_labels}") 
     
    # model_config = model.config
    # model_name_or_path = model_config.name_or_path
    # print("model_name_or_path = ", model_name_or_path)
    
    # if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    #     padding_side = "left"
    # else:
    #     padding_side = "right"
    
    # print("padding_side = ", padding_side)

    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)

    model, tokenizer = prepare_model_tokenizer(model_path)

    # train_bidirectional_prompt_tuning(model, tokenizer)
    
    
    
    
    # 加载数据集
    dataset_name = "race"

    dataset_path = Config["datasets"][dataset_name]
    ds = load_dataset_from_huggingface(dataset_path,"high")
    
    
    classes= get_classes_for_dataset(dataset_path)
    initialize_prefix_prompts(ds, tokenizer,20, 768, classes)