from transformers import BertForSequenceClassification
# from transformers import PromptForSequenceClassification  
from peft import PromptTuningInit, PromptTuningConfig  

import torch
import torch.nn as nn

# 初始化模型  
model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()  

K=5
# Prefix Prompt配置  
prefix_prompt_config = PromptTuningConfig(  
    prompt_length=K,  # K为前缀Prompt Tokens的数目  
    prompt_init_method=PromptTuningInit.TEXT,  # 初始化方法  
    prompt_init_text="..."  # 可选的初始化文本  
)  

# 后缀Prompt Tokens（使用聚类中心）  
def get_suffix_prompts(cluster_centers):  
    """  
    将聚类中心转换为可训练的Prompt参数  
    """  
    suffix_prompts = torch.tensor(cluster_centers, requires_grad=True).cuda()  
    return suffix_prompts  

# 将Prompt Tokens添加到模型中  
def add_prompts_to_model(model, prefix_config, suffix_prompts):  
    """  
    将前缀和后缀Prompt Tokens添加到模型中  
    """  
    # 前缀Prompt  
    model = PromptForSequenceClassification(model, prompt_config=prefix_config)  
    # 后缀Prompt（需要自定义实现）  
    model.suffix_prompts = suffix_prompts  
    return model  

# 示例用法  
if __name__ == "__main__":  
    K = 5  # 可根据需要设置  
    cluster_centers = ...  # 从K-Means聚类得到  
    suffix_prompts = get_suffix_prompts(cluster_centers)  
    model = add_prompts_to_model(model, prefix_prompt_config, suffix_prompts)  
