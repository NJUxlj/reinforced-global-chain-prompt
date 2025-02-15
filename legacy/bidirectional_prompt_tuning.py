from transformers import (
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

from data_utils.load import (
    preprocess_function_race_pt, 
    preprocess_function_race,
    load_dataset_from_huggingface,
    preprocess_race,
)

from datasets import (
    Dataset,
    load_dataset,
)
from config.config import Config

import torch
import torch.nn as nn
import numpy as np
import csv
import evaluate

device = Config['device']

# 初始化模型  
model_path = Config["models"]["bert-base-uncased"]["model_path"]
model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()  
tokenizer = AutoTokenizer.from_pretrained(model_path)


K=5
# 定义双向Prompt Tuning的参数  
num_prefix_tokens = 5   # 前缀Prompt Tokens的数量  
num_suffix_tokens = 5   # 后缀Prompt Tokens的数量  
embedding_size = model.get_input_embeddings().embedding_dim

# Prefix Prompt配置  
# prefix_prompt_config = PromptTuningConfig(  
#     prompt_length=K,  # K为前缀Prompt Tokens的数目  
#     prompt_init_method=PromptTuningInit.TEXT,  # 初始化方法  
#     prompt_init_text="..."  # 可选的初始化文本  
# )  


# 1. initialize the trainable prefix prompt embeddings
prefix_prompt_embeddings = torch.nn.Parameter(
    torch.rand(num_prefix_tokens, embedding_size,requires_grad=True, device= device),   # (num_prefix_tokens, embedding_size)
)

# 2. initialize the trainable suffix prompt embeddings
suffix_prompt_embeddings  = torch.nn.Parameter(  
    torch.rand(num_suffix_tokens, embedding_size,requires_grad=True, device= device),   # (num_suffix_tokens, embedding_size)
)


# 3. 修改模型的输入嵌入函数，添加前缀和后缀Prompt Tokens  
class BidirectionalPromptModel(torch.nn.Module):  
    def __init__(self, model, prefix_embeddings, suffix_embeddings):  
        super(BidirectionalPromptModel, self).__init__()  
        self.model = model  
        self.prefix_embeddings = prefix_embeddings  
        self.suffix_embeddings = suffix_embeddings  
        self.embedding_layer = self.model.get_input_embeddings()  
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):  
        # 原始输入嵌入
        print(f"input_ids.shape = {input_ids.shape}")
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
            prefix_mask = torch.ones(batch_size, num_prefix_tokens, device=device)  
            suffix_mask = torch.ones(batch_size, num_suffix_tokens, device=device)  

            # print(f"attention_mask.shape = {attention_mask.shape}")
            # print(f"prefix_mask.shape = {prefix_mask.shape}")
            # print(f"suffix_mask.shape = {suffix_mask.shape}")
            
            attention_mask = attention_mask.squeeze(1)
            attention_mask = torch.cat([prefix_mask, attention_mask, suffix_mask], dim=1)  # (4, 522)
            
            print(f"attention_mask.shape after concat = {attention_mask.shape}") 
        
        if token_type_ids is not None:  
            prefix_type_ids = torch.zeros(batch_size, num_prefix_tokens, device=device)
            suffix_type_ids = torch.zeros(batch_size, num_suffix_tokens, device=device)
            
            # print(f"token_type_ids.shape = {token_type_ids.shape}")
            # print(f"prefix_type_ids.shape = {prefix_type_ids.shape}")
            # print(f"suffix_type_ids.shape = {suffix_type_ids.shape}")
            
            token_type_ids = token_type_ids.squeeze(1)
            token_type_ids = torch.cat([prefix_type_ids, token_type_ids, suffix_type_ids], dim=1)
            
            token_type_ids = token_type_ids.long() # (4, 522)
            
            print(f"token_type_ids.shape after concat = {token_type_ids.shape}")

        
        # 调用原始模型的forward方法  
        outputs = self.model(  
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids,
            labels=labels  
        )  
        
        return outputs  

# 4. 创建带有双向Prompt的模型实例  
bidirectional_prompt_model = BidirectionalPromptModel(  
    model=model,  
    prefix_embeddings=prefix_prompt_embeddings,  
    suffix_embeddings=suffix_prompt_embeddings  
).to(device)  


# 后缀Prompt Tokens（使用聚类中心）  
def initialize_suffix_prompts(cluster_centers):  
    """  
    将聚类中心转换为可训练的Prompt参数  
    """  
    suffix_prompts = torch.tensor(cluster_centers, requires_grad=True).cuda()  
    return suffix_prompt_embeddings

def initialize_prefix_prompts(classification_tokens): 
    """
    将前缀Prompt Tokens初始化为一个随机向量
    """
    prefix_prompt_embeddings = torch.nn.Parameter(torch.zeros(num_prefix_tokens, embedding_size,requires_grad=True, device= device),   # (num_prefix_tokens, embedding_size)
    )
    
    return prefix_prompt_embeddings

# # 将Prompt Tokens添加到模型中  
# def add_prompts_to_model(model, prefix_config, suffix_prompts):  
#     """  
#     将前缀和后缀Prompt Tokens添加到模型中  
#     """  
#     # 前缀Prompt  
#     model = PromptForSequenceClassification(model, prompt_config=prefix_config)  
#     # 后缀Prompt（需要自定义实现）  
#     model.suffix_prompts = suffix_prompts  
#     return model  

# 示例用法  
# if __name__ == "__main__":  
#     K = 5  # 可根据需要设置  
#     cluster_centers = ...  # 从K-Means聚类得到  
#     suffix_prompts = get_suffix_prompts(cluster_centers)  
#     model = add_prompts_to_model(model, prefix_prompt_config, suffix_prompts)  





# 5. 数据预处理函数  
def preprocess_function(examples):  
    """  
    对输入文本进行编码，不添加特殊的前缀和后缀Tokens。  
    前缀和后缀Prompt Tokens已在模型中处理。  
    """  
    # inputs.shape = (batch_size, )
    inputs = examples['context'] + " " + examples['question'] + " " + examples['choices']  

    # 这里inputs肯定超过512字符了， bert只能允许512， 因此就放不下prefix和suffix tokens, 
    # 因此我们这里必须先把input截断成刚好502
    
    # for text in inputs:
    #     if len(text) > 512:
    #         print("text is too long, truncate it to 512")
    #         text = text[:512]
    #     elif len(text) < 512:
    #         print("text is too short, pad it to 512")
            
    encoding = tokenizer(  
        inputs,  
        padding="max_length",  
        truncation=True,  
        max_length=502,  # 502 + 2x5 = 512, 刚好达到bert的需要  
        return_tensors='pt'  
    )  
    
    # 检查 inputs 是否包含 token_type_ids  
    if "token_type_ids" in encoding:  
        print("token_type_ids are supported and present in the input tensor.")  
    else:  
        print("token_type_ids are not present in the input tensor.")  
    
    # print("encoding['input_ids'].shape  =", encoding['input_ids'].shape)
    
    # 将标签映射为数字索引，假设标签已经是数字形式  
    encoding['labels'] = torch.tensor(examples['label'], dtype=torch.long)  
    return {k: v.to(device) for k, v in encoding.items()}  

# 6. 准备数据集  
# 假设您已经有了经过NER、AutoCoT和K-Means处理的数据集，名为processed_dataset  
# 数据集包含以下字段：'context', 'question', 'choices', 'label'  
# 将数据集转换为Dataset对象  
def prepare_dataset(examples):  
    return preprocess_function(examples)  

# 示例：创建一个虚拟的数据集（请替换为您的实际数据集）  
data = {  
    'context': ["This is a sample context."] * 10,  
    'question': ["What is the answer to this question?"] * 10,  
    'choices': ["A) Option A B) Option B C) Option C D) Option D"] * 10,  
    'label': [0] * 10  # 假设正确答案为第一个选项  
}  
dataset = Dataset.from_dict(data)  
encoded_dataset = dataset.map(prepare_dataset)  

# 划分训练集和验证集  
split_dataset = encoded_dataset.train_test_split(test_size=0.2)  
train_dataset = split_dataset['train']  
eval_dataset = split_dataset['test']  

# print("train[0] = ", train_dataset[0]['input_ids'])

# 7. 定义训练参数  
training_args = TrainingArguments(  
    output_dir=Config["output_dir"],  
    evaluation_strategy='epoch',  
    learning_rate=5e-5,  
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,  
    num_train_epochs=5,  
    weight_decay=0.01,  
    logging_dir=Config["logging_dir"],  
    logging_steps=5,
    fp16=True
)  

# 8. 定义评估指标  
def compute_metrics(eval_pred):  
    logits, labels = eval_pred  
    predictions = np.argmax(logits, axis=-1)  
    accuracy = (predictions == labels).mean()  
    return {'accuracy': accuracy}  

# 9. 创建Trainer实例  
trainer = Trainer(  
    model=bidirectional_prompt_model,  
    args=training_args,  
    train_dataset=train_dataset,  
    eval_dataset=eval_dataset,  
    tokenizer=tokenizer,  
    compute_metrics=compute_metrics,  
)  

# 10. 开始训练  
trainer.train()  

# 11. 评估模型  
eval_results = trainer.evaluate()  
print(f"Evaluation results: {eval_results}")  

# 12. 保存模型  
# trainer.save_model(Config['save_model_dir'])  
# torch.save(bidirectional_prompt_model.state_dict(), Config['save_model_dir']['bert-base-uncased']['bidirectional-prompt-tuning']['race'])  
