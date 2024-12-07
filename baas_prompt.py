from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    AutoModel,
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


from load import *

from utils import *

from causal_modeling import *

from components import (
    SentenceEncoder,
    BaasAttention
)

import time
import sys  
import os  
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

from config import *

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


import nltk  
# nltk.download('punkt') 
# nltk.download('punkt_tab') 
# nltk.download('stopwords')  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 




# device = Config['device']




@dataclass
class BaasPromptConfig:
    model_name: str = "bert-base-uncased"
    model_path: str = "bert-base-uncased"  # 预训练模型名称
    peft_method: str = "baas-prompt"
    auto_model_class:type = AutoModelForSequenceClassification # 对于类类型的字段，使用 type 作为类型注解
    dataset_name:str = "race" 
    prefix_length: int = 10                        # prefix-tuning的默认前缀长度  
    suffix_length: int = 10
    num_labels: int = 2                           # MCQA的选项数量 (A,B,C,D)  
    batch_size:int = 32
    num_epochs:int = 2
    dropout: float = 0.1                          # dropout率  
    max_seq_length: int = 512                         # 最大序列长度  
    learning_rate: float = 0.3                   # 前缀参数的学习率  
    model_learning_rate: float = 1e-5             # 模型参数的学习率（如果需要微调）  
    
    prefix_projection: bool = True               # 是否使用MLP投影前缀  
    prefix_hidden_size: int = 768               # MLP中的P_theta'  即，MLP输入的隐单元维度  huggingface 默认它==encoder_hidden_size
    encoder_hidden_size:int = 768   # bert的隐藏层维度
    
    warmup_steps: int = 500  # 添加预热步骤  
    weight_decay: float = 1e-5  # 添加权重衰减 
    beta1_decay:float = 0.9   #beta1: 一阶矩估计的指数衰减率（默认0.9）用于Adam优化器
    beta2_decay:float = 0.8   # 用于AdaFactor optimizer
    total_training_steps = 30000  # 总的训练步数
    early_stop_steps = 10
    optimizer_class:type = Adam 
    
    all_layers:bool = False  # 是否在所有层插入prefix, suffix
    is_prefix:bool = False   # 是否把continuous prompt tokens 作为前缀append在input上，还是类似prompt-tuning直接插入input
    
    gradient_accumulation_steps:int = 32  # 梯度累积的步数 = 目标批量大小(64) / 实际批量大小(2) = 32,  常见的选择是 8、16、32
    mixed_precision:str="fp16"  # 启用混合精度训练  
    max_grad_norm:float = 1.0  # 最大梯度范数   常用值：1.0, 5.0, 10.0
    norm_type:float = 2.0   # # 2.0表示L2范数，1.0表示L1范数
    
    seed:int=42
    
    debug:bool=False
    
    seq_cls_type:str='binary'
    
class BaasPromptEncoder(nn.Module):  
    def __init__(  
        self,   
        config: BaasPromptConfig,
        model_config:AutoConfig,
        # prefix_embeddings: torch.Tensor,  
        # suffix_embeddings: torch.Tensor,
        dropout: float = 0.1,
        device = Config.device
    ):  
        super().__init__()  
        self.hidden_size = config.prefix_hidden_size
        if self.hidden_size != model_config.hidden_size:
            raise ValueError(f"The hidden size of the model: {model_config.hidden_size} and the prefix projection layer: {config.prefix_hidden_size} must be the same.")
        self.num_layers = model_config.num_hidden_layers  
        self.n_head = model_config.num_attention_heads  
        self.prefix_projection = config.prefix_projection
        self.device = device
        self.config = config
        self.model_config = model_config
        self.lstm_layers = 2
        
        
        # prefix_embeds = None
        # suffix_embeds = None
        # # 1. 扩展batch维度  
        # if prefix_embeddings.dim()==2:
        #     prefix_embeds = prefix_embeddings.unsqueeze(0).expand(config.batch_size, -1, -1).to(self.device)
        # else:
        #     prefix_embeds = prefix_embeddings
        # if suffix_embeddings.dim()==2:
        #     suffix_embeds = suffix_embeddings.unsqueeze(0).expand(config.batch_size, -1, -1).to(self.device)
        # else:
        #     suffix_embeds = suffix_embeddings
            
        # # 直接赋值的tensor，除非特别设置requires_grad=True，否则不会更新
        # self.continuous_prompt_embeddings = torch.concat(
        #     [
        #         prefix_embeds,
        #         suffix_embeds,
        #     ], dim=1
        # )  # shape = (batch_size, prefix_length+suffix_length, hidden_size)
        
        # 底下的模块都是需要参数更新的
 
        # 2. 双向LSTM编码器  
        self.forward_lstm = nn.LSTM(  
            self.hidden_size,  
            self.hidden_size // 2,  
            num_layers=self.lstm_layers,  
            bidirectional=True,  
            batch_first=True  
        ).to(self.device)  
        self.backward_lstm = nn.LSTM(  
            self.hidden_size,  
            self.hidden_size // 2,  
            num_layers=self.lstm_layers, 
            bidirectional=True,  
            batch_first=True  
        ).to(self.device)  
        
        self.prefix_to_suffix_attention = MultiHeadAttention(  
            self.hidden_size, num_heads=8  
        ).to(self.device)  
        
        self.suffix_to_prefix_attention = MultiHeadAttention(  
            self.hidden_size, num_heads=8  
        ).to(self.device)  
        
        # 4. 自适应门控融合  
        self.prefix_gate = AdaptiveGate(self.hidden_size).to(self.device)  
        self.suffix_gate = AdaptiveGate(self.hidden_size).to(self.device)  
        
        # 5. 位置编码  
        self.prefix_pos_embedding = SinusoidalPositionalEmbedding(self.hidden_size).to(self.device)  
        self.suffix_pos_embedding = SinusoidalPositionalEmbedding(self.hidden_size).to(self.device)  
        
        # 6. 输出转换层  
        self.prefix_output_layer = OutputTransformation(self.hidden_size).to(self.device)  
        self.suffix_output_layer = OutputTransformation(self.hidden_size).to(self.device)  
        
        self.dropout = nn.Dropout(dropout).to(self.device)  
        self.layer_norm = nn.LayerNorm(self.hidden_size).to(self.device)  

    def forward(
            self,
            prefix_embeddings: torch.Tensor,
            suffix_embeddings: torch.Tensor,
        )->Tuple[torch.Tensor, torch.Tensor]:  
        '''
        Args:
            prefix_embeddings, suffix_embeddings: shape = (seq_length, hidden_size)
            
        return  
            prefix_output, suffix_output   shape = (batch_size, seq_length, hidden_size)
        
        '''
        
        
        prefix_embeds = None
        suffix_embeds = None
        # 扩展batch维度  
        if prefix_embeddings.dim()==2:
            prefix_embeds = prefix_embeddings.unsqueeze(0).expand(config.batch_size, -1, -1).to(self.device)
        else:
            prefix_embeds = prefix_embeddings
        if suffix_embeddings.dim()==2:
            suffix_embeds = suffix_embeddings.unsqueeze(0).expand(config.batch_size, -1, -1).to(self.device)
        else:
            suffix_embeds = suffix_embeddings
            
        # 直接赋值的tensor，除非特别设置requires_grad=True，否则不会更新
        self.continuous_prompt_embeddings = torch.concat(
            [
                prefix_embeds,
                suffix_embeds,
            ], dim=1
        )  # shape = (batch_size, prefix_length+suffix_length, hidden_size)
            
        # 不需要过prompt_encoder的情况, 只训练continuous_prompt_embeddings， 其余的encoder都用不到(不输入且不更新)
        if not self.prefix_projection:
            prefix_embeds,suffix_embeds = torch.split(
                self.continuous_prompt_embeddings.clone(),
                [self.config.prefix_length, self.config.suffix_length],
                dim=1
            )
            return prefix_embeds, suffix_embeds
        
        # shape = (batch_size, prefix_length+suffix_length, hidden_size)
        # all_embeddings = self.embedding(self.continuous_prompt_embeddings)
        
        prefix_embeds,suffix_embeds = torch.split(
                self.continuous_prompt_embeddings.clone(),
                [self.config.prefix_length, self.config.suffix_length],
                dim=1
            )
        
        # 添加位置编码  
        prefix_embeds = self.prefix_pos_embedding(prefix_embeds)  
        suffix_embeds = self.suffix_pos_embedding(suffix_embeds)  
        
        # 双向LSTM编码  
        # 前向处理  
        prefix_forward, _ = self.forward_lstm(prefix_embeds)  
        suffix_forward, _ = self.forward_lstm(suffix_embeds)  
        
        # 反向处理  
        prefix_backward, _ = self.backward_lstm(torch.flip(prefix_embeds, [1]))  
        suffix_backward, _ = self.backward_lstm(torch.flip(suffix_embeds, [1]))  
        prefix_backward = torch.flip(prefix_backward, [1])  
        suffix_backward = torch.flip(suffix_backward, [1])  
        
        # 融合双向表示  
        prefix_bidirectional = self.prefix_gate(prefix_forward, prefix_backward)  
        suffix_bidirectional = self.suffix_gate(suffix_forward, suffix_backward)  
        
        # 交互注意力  
        prefix_attended,_ = self.prefix_to_suffix_attention(  
            prefix_bidirectional, suffix_bidirectional, suffix_bidirectional  
        )  
        suffix_attended,_ = self.suffix_to_prefix_attention(  
            suffix_bidirectional, prefix_bidirectional, prefix_bidirectional  
        )  
        
        # 残差连接和归一化  
        prefix_output = self.layer_norm(prefix_bidirectional + prefix_attended)  
        suffix_output = self.layer_norm(suffix_bidirectional + suffix_attended)  
        
        # 最终转换  
        prefix_output = self.prefix_output_layer(prefix_output)  
        suffix_output = self.suffix_output_layer(suffix_output)  
        
        return prefix_output, suffix_output 
    


class BassPromptModel(torch.nn.Module):  
    def __init__(
        self, 
        model, 
        tokenizer, 
        config:BaasPromptConfig, 
        chain_encode_args:ChainEncodingArguments=None, 
        device = Config.device,
        debug:bool=False
        ):  
        super(BassPromptModel, self).__init__()  
        self.device = device
        self.model = model.to(self.device)
        self.config:BaasPromptConfig = config
        self.base_model = get_base_model_using_model(self.model).to(self.device)
        self.model_config = self.model.config
        self.model_type = self.model_config.model_type 
        
        self.tokenizer = tokenizer
        self.hidden_size = config.prefix_hidden_size # bert's hidden size
        self.prefix_hidden_size = config.prefix_hidden_size # P_theta' in MLP reparameterization
        self.num_prefix_tokens = config.prefix_length
        self.num_suffix_tokens = config.suffix_length
        
        self.all_layers = config.all_layers # 是否插入所有层
        self.is_prefix = config.is_prefix # 是前缀还是插入？
        self.num_layers = self.model_config.num_hidden_layers
        self.debug=debug
        
        
        d_ff = self.hidden_size*4
        self.rollback_decoder = RollbackDecoderWithHead(
            model = self.model,
            d_model=self.hidden_size,
            d_ff=d_ff,
            num_heads=8,
            debug = False,
            device = self.device
        ).to(self.device)
        
        if chain_encode_args is None:
            self.chain_encode_args = ChainEncodingArguments(
                dataset=config.dataset_name,
                hidden_size=config.prefix_hidden_size, # 这个hidden_size最终会传给encode cot chain用到的 sentence transformer
                output_dir="./autocot/experiment/race",
                embedding_dir = "./autocot/embeddings/race",
                context_dir = "./autocot/context/race"
            )
        else:
            self.chain_encode_args = chain_encode_args 
            
        self.prefix_embeddings:torch.Tensor = self.initialize_prefix_prompts(
            dataset_path=get_dataset_path_by_name(config.dataset_name),
            tokenizer=self.tokenizer,
            model=self.model,
            hidden_size = self.hidden_size,
            config = config,
            classes_initiate_method = "cluster",
            num_topics = self.num_prefix_tokens,
            max_length = config.max_seq_length
        )
        
        self.suffix_embeddings:torch.Tensor = self.initialize_suffix_prompts() #shape =  (seq_length, hidden_size)
        
        # self.prefix_embeddings.requires_grad = True
        # self.suffix_embeddings.requires_grad = True
        self.prefix_embeddings = nn.Parameter(  
            self.prefix_embeddings.detach().clone()  # 创建新的叶子节点  
        ).to(self.device)  
        
        self.suffix_embeddings = nn.Parameter(  
            self.suffix_embeddings.detach().clone()  # 创建新的叶子节点  
        ).to(self.device) 
        
        # 整个模型需要更新的参数都在这里
        self.prompt_encoder = BaasPromptEncoder(
            self.config, 
            self.model_config, 
            # self.prefix_embeddings,
            # self.suffix_embeddings,
            device = self.device
            ).to(self.device)
          
        # self.prefix_embeddings, self.suffix_embeddings = self.prompt_encoder.forward()  
        # shape = (batch_size, seq_length, hidden_size), 等下会在_init_函数中把他扩展成真实的batch_size
        

        self.embedding_layer = self.model.get_input_embeddings().to(self.device)  # 获取词嵌入层
        
        # 这个可以更新
        self.classifier = get_classifier_from_model(self.model).to(self.device)
        
        self.disable_grad_calc() # 禁用梯度计算
        
        self.init_trainable_parameters()
        
    def get_past_key_values(
        self, 
        prefix_embeddings:torch.Tensor,
        input_embeddings:torch.Tensor,
        suffix_embeddings:torch.Tensor,
        batch_size=1,
        ):
        '''
        作用：
            把prefix_embeddings 和 suffix_embeddings转换bert每一层中的K,V矩阵能用的格式
        '''
        past_embeddings = torch.concat([prefix_embeddings, input_embeddings, suffix_embeddings], dim=1) # shape = (batch_size, max_length, hidden_size)


        past_embeddings = past_embeddings.unsqueeze(-1).expand(batch_size, self.config.max_seq_length, self.hidden_size, self.num_layers*2)

        return past_embeddings
        
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None,
        position_ids: Optional[torch.Tensor]=None, 
        labels:Optional[torch.Tensor] = None,
        head_mask=None,
        # inputs_embeds=None,
        output_attentions:bool=False,
        output_hidden_states:bool=False,
        return_dict:bool=True, # 返回 SequenceClassifierOutput 对像
        ):  
        # 原始输入嵌入
        if self.debug:
            print(f"input_ids.shape = {input_ids.shape}")  # List[List[int]]
            print(f"attention_mask.shape = {attention_mask.shape}")  # List[List[int]]
            print(f"max position embedding: {self.config.max_seq_length}") 
            
        sequence_length = self.num_prefix_tokens + attention_mask.size(1) + self.num_suffix_tokens  
        if sequence_length > self.config.max_seq_length:  
            raise ValueError(  
                f"Combined sequence length ({sequence_length}) exceeds model's maximum "  
                f"position embeddings ({self.base_model.config.max_position_embeddings})\n"  
                f"Components: prefix({self.num_prefix_tokens}) + "  
                f"input({attention_mask.size(1)}) + "  
                f"suffix({self.num_suffix_tokens})"  
            )  
        '''
        RoBERTa 是支持直接使用 inputs_embeds 的，问题在于位置编码（position embeddings）。

            看错误堆栈，问题出在 position_embeddings 这一层，这说明：

            当使用 inputs_embeds 时，我们还需要正确处理 position_ids
            默认的 position_ids 可能超出了范围限制
        '''
        if position_ids==None:
            position_ids = torch.arange(  
                0, sequence_length,   
                dtype=torch.long,   
                device=self.device  
            ).expand(self.config.batch_size, -1)  # [batch_size, seq_length]  
            
        # 确保不会重复计算或重用张量
        with torch.set_grad_enabled(True):
            inputs_embeds:torch.Tensor = self.embedding_layer(input_ids)  
            
            # shape = (batch_size, seq_length, hidden_size)
            prefix_embeds, suffix_embeds = self.prompt_encoder.forward(
                prefix_embeddings=self.prefix_embeddings,
                suffix_embeddings=self.suffix_embeddings
            ) 
            
            # 每轮反向传播后，self.prefix_embeddings, self.suffix_embeddings 都会更新
            # 然后这俩货在下一轮传入同样更新过的prompt-encoder
            
            # 使用RollbackEncoder对suffix_embeds进行rollbcak half N, residual connection
            suffix_embeds= suffix_embeds + self.rollback_half_extend(suffix_embeds, self.rollback_decoder)
            
            
            batch_size = inputs_embeds.size(0)  
            input_size = inputs_embeds.size(1)
            
            assert batch_size == self.config.batch_size, "input embedding's batch_size must be equal to config.batch_size"
            
        
            # past_key_values = self.get_past_key_values(
            #     prefix_embeds,
            #     inputs_embeds,
            #     suffix_embeds,
            #     batch_size) if config.all_layers else None
            
            
            if self.all_layers:
                # 插入self.base_model的每一层
                self.add_to_all_layers(
                    prefix_embeds,
                    suffix_embeds,
                    is_prefix=self.is_prefix
                    )
            
            if self.debug:
                print("************* 前缀， 中缀， 后缀 拼接前：*************")
                print(f"prefix.shape = {prefix_embeds.shape}")
                print(f"suffix.shape = {suffix_embeds.shape}")
                print(f"inputs_embeds.shape = {inputs_embeds.shape}")
                print("**************************************************")
                print()
            # 拼接前缀、原始输入和后缀嵌入  
            
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds, suffix_embeds], dim=1)  # (4, 512, 768)
            
            # 调整attention_mask  
            if attention_mask is not None:  
                prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=self.device)  
                suffix_mask = torch.ones(batch_size, self.num_suffix_tokens, device=self.device)  

                
                prefix_mask = prefix_mask.to(self.device)  
                attention_mask = attention_mask.to(self.device)  
                suffix_mask = suffix_mask.to(self.device) 
                 
                if self.debug:  
                    debug_cuda_sync("Attention Mask Device transfer in function `forward`")  

                if self.debug:
                    print("********************* attention_mask.shape ******************************")
                    print(f"attention_mask.shape = {attention_mask.shape}")
                    print(f"prefix_mask.shape = {prefix_mask.shape}")
                    print(f"suffix_mask.shape = {suffix_mask.shape}")
                    
                # # 如果attention_mask是4D，保持原样  
                # # 如果是2D或3D，扩展到4D  
                # if attention_mask.dim() == 2:  
                #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  
                # elif attention_mask.dim() == 3:  
                #     attention_mask = attention_mask.unsqueeze(1)  

                # # 同样扩展prefix和suffix mask的维度  
                # prefix_mask = prefix_mask.unsqueeze(1).unsqueeze(2)  
                # suffix_mask = suffix_mask.unsqueeze(1).unsqueeze(2)  
                
                # attention_mask = attention_mask.squeeze(1)
                attention_mask = torch.cat([prefix_mask, attention_mask, suffix_mask], dim=-1)  # (4, 522)
                
                if self.debug:  
                    debug_cuda_sync("attention_mask Concatenation")  
                
                if self.debug:
                    print("\n************* after unsqueeze, the attention mask shape become:***************")
                    print(f"prefix_mask.shape after unsqueeze = {prefix_mask.shape}")
                    print(f"suffix_mask.shape after unsqueeze = {suffix_mask.shape}")
                    print(f"attention_mask.shape after concat = {attention_mask.shape}") 
                    
                expected_seq_length = self.num_prefix_tokens + input_size + self.num_suffix_tokens  
                assert attention_mask.size(-1) == expected_seq_length, \
                    f"Attention mask sequence length {attention_mask.size(-1)} " \
                    f"doesn't match expected length {expected_seq_length}"  
                    
                if self.debug:
                    debug_cuda_sync("Final validation for attention mask")  
            
            if token_type_ids is not None:  
                prefix_type_ids = torch.zeros(batch_size, self.num_prefix_tokens, device=self.device)
                suffix_type_ids = torch.zeros(batch_size, self.num_suffix_tokens, device=self.device)
                
                if self.debug:
                    print("********************* token_type_ids.shape ******************************")
                    print(f"token_type_ids.shape = {token_type_ids.shape}")
                    print(f"prefix_type_ids.shape = {prefix_type_ids.shape}")
                    print(f"suffix_type_ids.shape = {suffix_type_ids.shape}")
                
                # token_type_ids = token_type_ids.squeeze(1)
                token_type_ids = torch.cat([prefix_type_ids, token_type_ids, suffix_type_ids], dim=1)
                
                token_type_ids = token_type_ids.long() # (4, 522)
                
                if self.debug:
                    print(f"token_type_ids.shape after concat = {token_type_ids.shape}")
                    print("**************************************************************************\n")

            # time.sleep(10000)
            
            # 调用原始模型的forward方法  
            outputs = self.base_model(  
                # input_ids=input_ids,                 
                inputs_embeds=inputs_embeds,  # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是从input_ids重新计算
                attention_mask=attention_mask,  
                token_type_ids=token_type_ids if self.model_type == "bert" or self.model_type == "roberta" else None,  
                position_ids=position_ids if self.model_type == "bert" or self.model_type == "roberta" else None,
                # labels=labels,
                head_mask=head_mask,
                output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                return_dict=return_dict,
                # past_key_values.shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
                # past_key_values=past_key_values if config.all_layers else None, 
            )   
            
            # pooled_output = outputs[1] # shape = (batch_size, hidden_size)
            # cls_token = outputs.hidden_states[-1][:, 0, :] # shape = (batch_size, hidden_size)
            # cls_token = outputs.last_hidden_state[:, 0, :] # shape = (batch_size, hidden_size)

            if hasattr(self.classifier, 'dense') or hasattr(self.classifier, 'out_proj'):
                # it means that the classifier is a RobertaClassificationHead
                logits:torch.Tensor = self.classifier(outputs.last_hidden_state) # shape = (batch_size, num_labels)
            else:
                logits = self.classifier(outputs.last_hidden_state[:,0,:]) # shape = (batch_size, num_labels)
            
            if self.debug:
                print("************* BaasPromptModel 中的 base_model 输出：*************")
                print("logits.shape = ",logits.shape)
                # print("logits = ",logits)
                print("**********************************************************\n")
            
            
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
                hidden_states=outputs.hidden_states if hasattr(outputs,"hidden_states") else None,
                attentions=outputs.attentions if hasattr(outputs,"attentions") else None,
            ) 

    def print_trainable_parameters(self):  
        """
        print trainable parameters' number and ratio
        
        only prefix, suffix, base model parameters are calculated
        
        """  
        trainable_params = 0  
        all_params = 0  
        
        for name, param in self.base_model.named_parameters():
            num_params = param.numel()
            all_params+=num_params
            
        for name, param in self.named_parameters():  
            num_params = param.numel()  
            if param.requires_grad and (
                                        "prompt_encoder" in name or 
                                        "rollback_decoder" in name or 
                                        "classifier" in name or 
                                        "prefix_embeddings" in name or "suffix_embeddings" in name):  
                trainable_params += num_params
                all_params += num_params  
                  
        print(f"trainable param number: {trainable_params}")  
        print(f"total param number: {all_params}")  
        print(f"trainable param ratio: {100 * trainable_params / all_params:.2f}%")  

    def disable_grad_calc(self):
        """disable grad calc for base model parameters"""
        for name, param in self.named_parameters():
            if "prompt_encoder" in name:
                param.requires_grad = True
            elif "rollback_decoder" in name:
                param.requires_grad = False
            elif "classifier" in name:
                param.requires_grad = True
            elif "prefix_embeddings" in name:
                param.requires_grad = True
            elif "suffix_embeddings" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.print_trainable_parameters()
    
    def init_trainable_parameters(self):
        
        print("********************** Initializing trainable parameters: **********************")
        for name, param in self.named_parameters():
            if "prompt_encoder" in name:
                if 'weight' in name:  
                    if len(param.shape) >= 2:  
                        # 权重矩阵使用 xavier_uniform  
                        torch.nn.init.xavier_uniform_(param)  
                    else:  
                        # 1维权重使用正态分布  
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)   
                elif 'bias' in name:  
                    torch.nn.init.zeros_(param)
                elif 'embeddings' in name:  
                    # 嵌入层使用正态分布  
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)  
                else:  
                    # 其他参数使用均匀分布  
                    torch.nn.init.uniform_(param, -0.1, 0.1) 
                print(f"Initialized {name} with shape {param.shape}")  
            
            elif "rollback_decoder" in name:
                if 'weight' in name:  
                    if len(param.shape) >= 2:  
                        # 权重矩阵使用 xavier_uniform  
                        torch.nn.init.xavier_uniform_(param)  
                    else:  
                        # 1维权重使用正态分布  
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)   
                elif 'bias' in name:  
                    torch.nn.init.zeros_(param)
                elif 'embeddings' in name:  
                    # 嵌入层使用正态分布  
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)  
                else:  
                    # 其他参数使用均匀分布  
                    torch.nn.init.uniform_(param, -0.1, 0.1) 
                print(f"Initialized {name} with shape {param.shape}")
                
            elif "classifier" in name:
                if 'weight' in name:  
                    if len(param.shape) >= 2:  
                        # 权重矩阵使用 xavier_uniform  
                        torch.nn.init.xavier_uniform_(param)  
                    else:  
                        # 1维权重使用正态分布  
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)   
                elif 'bias' in name:  
                    torch.nn.init.zeros_(param)
                elif 'embeddings' in name:  
                    # 嵌入层使用正态分布  
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)  
                else:  
                    # 其他参数使用均匀分布  
                    torch.nn.init.uniform_(param, -0.1, 0.1) 
                print(f"Initialized {name} with shape {param.shape}")
                
        print("***********************************************\n\n\n")
                    
                    

    def initialize_suffix_prompts(self)->torch.Tensor:  
        """  
        use Auto-CoT reasoning steps as the suffix prompts
        
        return 
        
            shape = (num_suffix_tokens, embedding_size)
        """  
        # 假设我们已经有AutoCoT生成的推理步骤steps，并将其保存为文本列表
        context = rollback_one_step_extend(
            self.num_suffix_tokens,
            args = self.chain_encode_args,
            model = self.model,
            decoder = self.rollback_decoder
            )
        
        suffix_embeddings = context
        
        
        return suffix_embeddings
    

    def initialize_prefix_prompts(
        self, 
        dataset_path,
        model,
        tokenizer,
        hidden_size, 
        config:BaasPromptConfig,
        classes_initiate_method = "cluster",
        num_topics=5,
        max_length=512,
        )->torch.Tensor: 
        """ 
        use the class labels to initialize the prefix tokens
        
        return tensor shape = (num_prefix_tokens, embedding_size)
        """
        # model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
        
        class_embeddings = None
        if classes_initiate_method == "normal":
            class_embeddings = get_classes_for_dataset(dataset_path, model, tokenizer, embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
        elif classes_initiate_method == "cluster":
            class_embeddings:torch.Tensor = get_classes_by_clustering(dataset_path, model, tokenizer, config=config ,embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
        elif classes_initiate_method == "lda":
            class_embeddings = get_classes_by_lda(dataset_path, model, tokenizer, embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
        else:
            raise ValueError("Invalid classes_initiate_method, Please choose from ['normal', 'cluster', 'lda']")

        
        prefix_embeddings = torch.zeros(self.num_prefix_tokens, hidden_size, device=self.device)
        
        for i in range(self.num_prefix_tokens):  
            prefix_embeddings[i] = class_embeddings[i]
        
        # requires_grad 统一到外面进行处理
        # prefix_prompt_embeddings = torch.nn.Parameter(prefix_embeddings, requires_grad=True)   # (num_prefix_tokens, embedding_size)

        return prefix_embeddings

    
    def rollback_half_extend(self, suffix_embeds:torch.Tensor, rollback_decoder):
        '''
        suffix_embeds.shape = (batch_size, num_suffix_tokens, hidden_size)
        '''
        
        suffix_embeddings = suffix_embeds.detach().clone().to(self.device)  
        if self.debug:
            print(f"压缩推理链 suffix embedding steps from {self.num_suffix_tokens} to {self.num_suffix_tokens//2}")
        source_length = self.num_suffix_tokens//2
        suffix_embeddings = suffix_embeddings[:,:source_length,:] # rollback N/2 step
        
        # use multihead-attention to generate new steps using causual language modeling
        for i in range(source_length, self.num_suffix_tokens):
            # 生成新的推理步骤
            token_id, new_step_embedding = self.generate_new_step(suffix_embeddings, rollback_decoder) # shape = [hidden_size]
            new_step_embedding = new_step_embedding.unsqueeze(0).unsqueeze(0).expand(len(suffix_embeddings),1,self.hidden_size) # [1, hidden_dim]
            suffix_embeddings = torch.cat([suffix_embeddings, new_step_embedding], dim=1)

        return suffix_embeddings

    def generate_new_step(self, context:torch.Tensor, rollback_decoder:RollbackDecoderWithHead, temperature=0.7):
        """生成下一个token并返回其embedding  
        
        Args:  
            context: 形状为 [seq_len, hidden_dim] 的上下文张量  
            model: 带有输出头的解码器模型  
            temperature: 采样温度  
            : 词表大小（默认使用BERT的词表大小）  
        
        Returns:  
            tuple: (next_token_id, next_token_embedding)  
                - next_token_id: 下一个token的ID  
                - next_token_embedding: 下一个token的embedding向量 [hidden_dim]  
        """  

        # 创建因果掩码  
        seq_len = context.size(0)  
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device), diagonal=1).bool()  
        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]  
        
        '''
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()：
            创建一个上三角矩阵，其中主对角线以上的元素为 1，主对角线及以下的元素为 0。
            torch.triu 函数用于生成上三角矩阵，diagonal=1 表示从主对角线开始的位置。bool() 函数将结果转换为布尔类型。

        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(1)：
            对因果掩码进行取反操作，即将 1 变为 0，将 0 变为 1。
            然后，使用 unsqueeze(0) 和 unsqueeze(1) 在维度 0 和 1 上分别添加一个维度，
            使得因果掩码的形状变为 [1, 1, seq_len, seq_len]。
        '''
        
        
        # 将context扩展为batch维度
        if context.dim()==2:
            context = context.unsqueeze(0)  # [1, seq_len, hidden_dim] 
        
        if context.dim()==3:
            context = context[:1] # # 保留第一个batch 
        
        with torch.no_grad():  
            # 通过解码器生成新的隐藏状态  
            next_token_logits  = rollback_decoder.forward(context, causal_mask)  # shape = [1, vocab_size]
            
            if self.debug:
                print("next_token_logits.shape = ", next_token_logits.shape)
                print("next_token_logits = ", next_token_logits)
            
            # 应用温度缩放  
            # # temperature 影响概率分布的"锐利度", <1 使分布更尖锐，>1 使分布更平缓    
            if temperature != 1.0:  
                next_token_logits = next_token_logits / temperature  
            
            # 计算概率分布  
            probs = F.softmax(next_token_logits, dim=-1)  
            
            if self.debug:
                print("probs.shape = ", probs.shape)
                print("probs = ", probs)    
            # 采样下一个token  
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1] 
            
            
            '''
            # multinomial 根据概率分布进行采样  
            # - probs: 概率分布 [1, vocab_size]  
            # - num_samples=1: 采样一个token  
            # - 返回的是词表中的索引（token ID）  

            # 例如，对于概率分布 [0.01, 0.31, 0.42, 0.26]：  
            # - 42% 的概率选择索引 2  
            # - 31% 的概率选择索引 1  
            # - 26% 的概率选择索引 3  
            # - 1% 的概率选择索引 0  
            
            '''
            

            # 获取token的embedding  
            next_token_embedding = rollback_decoder.embed_tokens(next_token)  # [1, 1, hidden_dim]  
            next_token_embedding = next_token_embedding.squeeze().to(self.device)  # [hidden_dim]  
            
            if self.debug:
                print("next_token_embedding.shape = ", next_token_embedding.shape)
                print("next_token_embedding = ", next_token_embedding)

        # return new_step.squeeze(0)  # [hidden_dim]  
        return next_token.squeeze().item(), next_token_embedding
    
    def add_to_all_layers(
        self,
        prefix_embeddings:torch.Tensor,
        suffix_embeddings:torch.Tensor,
        is_prefix=False,
        ):
        '''
        prefix_embeddings.shape = (batch_size, prefix_length, hidden_size)
        suffix_embedding.shape = (batch_size, suffix_length, hidden_size)
        '''
        # 将prefix 和 suffix 更新到Bert中的每一层
        for i, layer in enumerate(self.base_model.encoder.layer):  
            layer.attention.self = BaasAttention(  
                config=self.base_model.config,  
                layer_idx=i,  
                fixed_svd=None,
                prefix_embeddings=prefix_embeddings,
                suffix_embeddings=suffix_embeddings,
                is_prefix=is_prefix
            )  

def reformat_input(config:BaasPromptConfig, tokenizer, reformat_type = "normal"):
    '''
       This function will be used when generating class labels for each sample in the dataset.
        
       根据数据集的格式将问题格式化为相应的字符串
        e.g. if dataset = race: "Article: ... Question: ... Options: ... Answer:"
        
        
        reformat_type: "normal" or "lda" or "cluster":
        
            "normal": use the get_classes_for_dataset() to get class labels
            "lda": use the get_classes_lda() to get class labels
            "cluster": use the get_classes_cluster() to get class labels
    '''
    
    reformat_dict = {
        "normal": "get_classes_for_dataset()",
        "lda": "get_classes_by_lda()",
        "cluster": "get_classes_by_cluster()"
    }
    
    input_key = 'input'
    
    max_length = config.max_seq_length
    
    print(f"reformat input using reformat_type = {reformat_dict[reformat_type]}")
    wrapper = McqDatasetWrapper()
    dataset_configs = wrapper.dataset_configs
    
    if config.dataset_name == 'race' or config.dataset_name == 'sciq' \
        or config.dataset_name == 'dream' or config.dataset_name == 'commonsense_qa':
        
        
        article_key = dataset_configs[config.dataset_name].article_key
        question_key = dataset_configs[config.dataset_name].question_key
        options_key = dataset_configs[config.dataset_name].options_key
        label_key = dataset_configs[config.dataset_name].label_key
        
        
        
        if reformat_type == "normal": 
            ds,_ = wrapper.load_mcq_dataset(config.dataset_name) 
            # processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            train_ds = ds["train"]
            
        elif reformat_type == "lda":
            ds,_ = wrapper.load_mcq_dataset(config.dataset_name)
            
            processed_ds = ds.map(
                lambda examples: {
                    input_key: [f"{article_key}:{examples[article_key][index]}\n\n{question_key}:{examples[question_key][index]}\n\n \
                                            {options_key}:{examples[options_key][index]}\n\n{label_key}:" for index, x in enumerate(examples[article_key])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=[article_key, question_key, options_key],
                load_from_cache_file=False,
                desc=f"Running reformat function's mapping on dataset {config.dataset_name}",
            )
            
            # transfer the dataset into List[str]
            processed_ds = [text for text in processed_ds['train']['input']]  
            
            
            train_ds: List[List[str]] = []
            for text in processed_ds:
                text = text.lower()
                tokens = word_tokenize(text)
                # remove stopwords and non-alphabetic characters
                tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                train_ds.append(tokens)
            
        elif reformat_type == "cluster":
            # processed_ds = preprocess_dataset_peft(dataset_name, max_length = max_length)
            # train_ds = processed_ds["train"]
            ds,_ = wrapper.load_mcq_dataset(config.dataset_name)
            
            # print("load_mcq_dataset ds[0]", ds[0])
            print("type(ds) = ",type(ds))
            train_ds = ds["train"]
            
            train_ds = train_ds.map(
                lambda examples: {
                    input_key: [f"{article_key}:{examples[article_key][index]}\n\n{question_key}:{examples[question_key][index]}\n\n \
                                            {options_key}:{examples[options_key][index]}\n\n{label_key}:" for index, x in enumerate(examples[article_key])]  
                },
                batched=True,
                num_proc=NUM_CPU_PROCESSES,
                remove_columns=[article_key, question_key, options_key],
                load_from_cache_file=False,
                desc=f"Running reformat function's mapping on dataset {config.dataset_name}",
            )
            
            if input_key not in train_ds.column_names:  
                raise KeyError(  
                    f"Failed to create 'input' column. "  
                    f"Current columns: {train_ds.column_names}"  
                )  
            # train_ds = processed_ds["train"]
        
        else:
            raise ValueError("Invalid reformat_type, please choose from 'normal', 'lda', 'cluster")
        
    else:
        raise ValueError(f"dataset_name: {config.dataset_name} not supported, we can not reformat dataset using a wrong name, please change another in [race, sciq, commonsense_qa, dream]")
    
    return train_ds, input_key




def get_classes_for_dataset(dataset_path, model, tokenizer, embedding_size, num_topics = 5, K=5, max_length=512)->List[torch.Tensor]:
    '''
    get the label collection of some specific dataset
        
    Args:
        dataset_path: the path of dataset, should be in [race, race-m, race-h, multirc, arc]
        K: number of classes in the dataset
        model: we need to get the embedding weight to calc cosine similarity
        
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
    
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
    
    print("get class labels by TF-IQF-AS (normal method) ~~~")
    
    classes = []
    model = model.eval()
    device = Config['device']
    model.to(device)
    
    vocab = tokenizer.get_vocab()
    inverse_vocab = {v:k for k,v in vocab.items()} # index-token mapping
    
    all_pooled_embeddings = [] # store the pooled embeddings of every 1000 examples
    
    vocab_size = tokenizer.vocab_size
    # count the word frequency of each word in vocab on the training set
    word_freq = {token_id:0 for token_id in range(vocab_size)}
    
    
    train_ds = reformat_input(dataset_path, tokenizer, max_length=max_length)
    
    train_data_loader = DataLoader(train_ds, batch_size=1000, 
                                   collate_fn=default_data_collator, 
                                   num_workers=0, # use as your need
                                   shuffle=False)
    
    # record whether this token occurs in this example or not, 1 for yes, 0 for no
    word_occurence_per_example:Dict[int, Dict[int, int]] = {
                                                                token_id:{
                                                                    example_id:0 
                                                                        for example_id in range(len(train_ds))
                                                                    } 
                                                                for token_id in range(vocab_size)
                                                            }

    
    with torch.no_grad():
        global_step = 0
        for index, batch in enumerate(tqdm(train_data_loader)):
            # get the average embedding of each batch
            batch = {k:v.to(device) for k, v in batch.items()}
            
            input_ids = batch['input_ids']
            input_ids_np = input_ids.cpu().numpy()
            for ids in input_ids_np:  
                for id in ids:
                    id = int(id)  
                    word_freq[id] += 1
                    word_occurence_per_example[id][global_step] = 1
                
                global_step += 1
            
            # outputs = model(**batch)
            embeddings = model.embeddings(input_ids) # shape = (batch_size, seq_len, hidden_size)
            # firstly, pooling on the seq_len dimension
            pooled_embedding = embeddings.mean(dim=1) # shape = (batch_size, hidden_size)
            # then, pooling on the batch_size dimension
            # all_pooled_embeddings.append(pooled_embedding.mean(dim=0).cpu()) # shape = (hidden_size)
            all_pooled_embeddings.append(pooled_embedding.mean(dim=0).unsqueeze(0)) # shape = (1, hidden_size)
            
    
    # print("all_pooled_embeddings = ", all_pooled_embeddings)
    print("=================================")
    print("all_pooled_embeddings[0].shape = ", all_pooled_embeddings[0].shape)
    print("all_pooled_embeddings[10].shape = ", all_pooled_embeddings[10].shape)
    print("all_pooled_embeddings[20].shape = ", all_pooled_embeddings[20].shape)
    
    
    print("all_pooled_embeddings.length = ", len(all_pooled_embeddings))
    print("len(train_data) = ", len(train_data_loader)) # 88
    print("len(train_data)//1000 = ", len(train_data_loader)//1000) # 0
    print("==================================")
    # vertically stack all the pooled embeddings into a matrix
    # "dim" points to the dimension of each pooled embedding
    pooled_embeddings_matrix = torch.cat(all_pooled_embeddings, dim=0) # shape = (len(train)//1000, hidden_size)

    num_pooled = len(train_ds)//1000
    if num_pooled == 0:
        num_pooled = 1
        
    vocab_size =  tokenizer.vocab_size
    token_embeddings = model.embeddings.word_embeddings.weight # [vocab_size, hidden_size]
    
    
    print("pooled embeddings matrix shape: ", pooled_embeddings_matrix.shape)
    print("token embeddings matrix shape: ", token_embeddings.shape)
    
    # 计算池化嵌入与词汇表中每个词嵌入的余弦相似度，并取平均值
    all_similarities_token_to_corpus = []
    for token_embedding in tqdm(token_embeddings):
        pooled_embedding = pooled_embedding.squeeze(0) # shape = (hidden_size)
        cosine_similarities = F.cosine_similarity(
            x1=token_embedding,
            x2 =pooled_embeddings_matrix,
            dim=1 # 代表我们在行的维度上进行计算
        ) # shape = (num_pooled)
        
        all_similarities_token_to_corpus.append(cosine_similarities.mean(dim=0).item())

    all_similarities_token_to_corpus = torch.tensor(all_similarities_token_to_corpus)
    print("all_similarities_token_to_corpus.shape = ", all_similarities_token_to_corpus.shape)
    
    # avg_similarities = cosine_similarities.mean(dim=0) # shape = (vocab_size)
    
    
    # transfer the word frequency to tensor
    freq_tensor = torch.zeros(vocab_size) 
    for token_id, freq in word_freq.items():  
        if token_id < vocab_size:  
            freq_tensor[token_id] = freq  
    
    # store the total occurence number of each token on all examples
    word_occurence_total = [0]*len(train_ds)
    
    # calculate the total occurence in all examples of each token
    for token_id in range(vocab_size):
        for example_id in range(len(train_ds)):
            word_occurence_total[token_id] += word_occurence_per_example[token_id][example_id]
    
    # calculate TF-IQF-VS score
    tf_iqf_vs_scores:torch.Tensor = all_similarities_token_to_corpus * freq_tensor * torch.tensor(word_occurence_total/len(train_ds))
    
    # filter special tokens
    filtered_scores = []
    for token_id, score in enumerate(tf_iqf_vs_scores):
        token = inverse_vocab[token_id]
        if token.strip() and not token.startswith("[") and token.isalpha() and token not in stop_words:
            filtered_scores.append([token_id, score])
    filtered_scores = torch.tensor(filtered_scores)
    
    # choose Top-K as class labels
    topk_scores, topk_indices = filtered_scores.topk(num_topics, dim=1)
    for idx in topk_indices:    
        classes.append((idx, inverse_vocab[idx]))  
    
    print("class labels are: ")
    for idx, label in classes:
        print(f"class {idx}: {label}")
    
    class_embeddings = []
    for index, _ in classes:
        class_embeddings.append(token_embeddings[index])
    return class_embeddings

def get_classes_by_clustering(
    dataset_path, 
    model: AutoModelForSequenceClassification, 
    tokenizer, 
    config:BaasPromptConfig,
    embedding_size,  # hidden_size
    num_topics=5,  
    max_length=512, 
    use_trained_embeddings=True,
    cache_dir='./class_label_cluster_embeddings'  # 存储embeddings的目录  
    )->List[torch.Tensor]:
    '''
    Args:
        num_topics: number of classes to be generated == num_suffix_tokens
        
        K : number of sub-classes to be generated
        
        cache_dir: 存储embeddings的目录  
        use_trained_embeddings: 是否使用已缓存的embeddings  
    
    return 
        class_embeddings: a list of tensor embeddings, each embedding corresponds to a label
    
    '''
    print(f"***************** Get Class Labels by Clustering ~~~~ *********************88")
    os.makedirs(cache_dir, exist_ok=True)
    device = Config.device
    classes = []
    model = model.to(device)
    
    
    # 生成唯一的缓存文件名（基于模型名称和数据集路径）  
    model_name = get_model_name_using_model(model)
    dataset_name = os.path.basename(dataset_path) 
    cache_filename = f"label_embeddings_{dataset_name}_{embedding_size}.pt"  
    cache_path = os.path.join(cache_dir, cache_filename) 
    
    
    
    # 直接加载最终结果，有的话直接返回
    final_label_embeddings_filename = f"final_label_embeddings_{dataset_name}_{embedding_size}.pt"
    final_label_metadata_filename = f"final_label_metadata_{dataset_name}_{embedding_size}.pt"
    final_label_embeddings_path = os.path.join(cache_dir, 'final_embeddings',final_label_embeddings_filename)
    final_label_metadata_path = os.path.join(cache_dir, 'final_embeddings', final_label_metadata_filename)
    
    
    final_metadata = {
        'dataset_path': dataset_path,  
        'model_name': model_name,  
        'embedding_size': embedding_size  
    }
    
    if os.path.exists(final_label_embeddings_path) and os.path.exists(final_label_metadata_path):

        # 验证metadata
        with open(final_label_metadata_path, 'r') as f:  
                cached_final_metadata = json.load(f)  
                
        if all(cached_final_metadata[k] == final_metadata[k] for k in final_metadata.keys() if k != 'max_length'):  
            print(f"Loading final label embeddings from {final_label_embeddings_path}")
            device,local_rank = setup_distributed()
            cluster_label_embeddings = torch.load(final_label_embeddings_path, map_location=device)  
            print(f"Loaded final label embeddings shape: {cluster_label_embeddings.shape}")
            return cluster_label_embeddings
        else:  
            print("============= cached final meta data ========================")
            print(cached_final_metadata)
            print()
            print("============== final meta data ================================")
            print(final_metadata)
            print()
            print("Sentence embedding's cache final metadata mismatch, recomputing final label embeddings for clustering...") 
            print()

    # print(f"Loading final label embeddings from {final_label_embeddings_path}")
    # cluster_label_embeddings = torch.load(final_label_embeddings_path)  
    # print(f"Loaded final label embeddings shape: {cluster_label_embeddings.shape}")
    # return cluster_label_embeddings
    
    
  
    # 保存数据集信息，用于验证sentence embedding的缓存是否匹配  
    metadata = {  
        'dataset_path': dataset_path,  
        'model_name': model_name,  
        'embedding_size': embedding_size  
    }  
    metadata_path = os.path.join(cache_dir, f"{cache_filename}_metadata_{embedding_size}.json")  
    
    embeddings = None

    # 如果启用缓存且缓存文件存在，尝试加载缓存的embeddings  
    if use_trained_embeddings and os.path.exists(cache_path) and os.path.exists(metadata_path):  
        try:
            # 首先验证metadata是否匹配  
            with open(metadata_path, 'r') as f:  
                cached_metadata = json.load(f)  
                
            if all(cached_metadata[k] == metadata[k] for k in metadata.keys() if k != 'max_length'):  
                print(f"Loading cached embeddings from {cache_path}")  
                # 使用torch.load加载缓存的embeddings  
                embeddings = torch.load(cache_path)  
                print(f"Loaded embeddings shape: {embeddings.shape}")
                # return process_embeddings(embeddings, num_topics, K)  # 假设有这个后处理函数[聚类逻辑]
                
            else:  
                print("============= cached meta data ========================")
                print(cached_metadata)
                print()
                print("============== meta data ================================")
                print(metadata)
                raise RuntimeError("Sentence embedding's cache metadata mismatch, recomputing embeddings for clustering...") 

        except (json.JSONDecodeError, FileNotFoundError, RuntimeError) as e:
            print(f"Error, when loading cache: {e}, meta data = {metadata}. Recomputing embeddings...")


        # embeddings = torch.load(cache_path)  
        
    # else:
    train_ds, input_key = reformat_input(config, tokenizer, reformat_type = 'cluster')
    train_ds: Dict[str,List[str]]
    
    # 验证列名是否存在  
    if input_key not in train_ds.column_names:  
        available_columns = train_ds.column_names  
        raise KeyError(  
            f"Column '{input_key}' not found in dataset. "  
            f"Available columns are: {available_columns}. "  
            f"Please check your dataset structure or specify the correct input_key."  
        )  
    
    sentences = train_ds[input_key]
    
    print(f"The training data is reformated to only one column {input_key}, now we get each example's embedding using the SentenceTransformer~~~")
    
    encoder = SentenceEncoder(
        hidden_size=embedding_size
    ).to(device)
    
    embeddings = encoder.encode(
        sentences=sentences
    ) # shape = (dataset_size, hidden_size)
            
    
    print("all_embeddings.shape = ", embeddings.shape)
        
    
    # 保存embeddings到缓存  
    print(f"Saving new embeddings to {cache_path}")  
    torch.save(embeddings, cache_path)  
    
    # 保存metadata  
    with open(metadata_path, 'w') as f:  
        json.dump(metadata, f) 
    
    
    if embeddings ==None:
        raise RuntimeError("Sentence embeddings for clustering is None, created failed, please check the code")

    # no matter how you get embeddings (cache/model infer), we need to convert it to numpy array for clustering
    embeddings = embeddings.cpu().numpy() 
             
    # clustering
    print(f"doing K-Means clustering, cluster number = {num_topics}")
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(embeddings)
    
    centroids = kmeans.cluster_centers_ # shape = ndarray(n_clusters, n_features)
    
    print("prepare candidate vocabulary...")  
    vocab = tokenizer.get_vocab()  
    inv_vocab = {v: k for k, v in vocab.items()}  # inverse to a id-to-token mapping

    # filter the vocabulary，we only keep the words that are not stop words, all alphabetic words, and not in special tokens [CLS]...
    candidate_tokens = []
    for token_id in inv_vocab:  
        token = inv_vocab[token_id]  
        # skip special tokens, like: [CLS], [SEP], [PAD], [MASK], and sub-words (begining with ##)  
        if token.startswith('[') or token.startswith('##'):  
            continue  
        if token.isalpha() and token not in stop_words:  
            candidate_tokens.append(token)  

    print(f"candidate token numbers: {len(candidate_tokens)}")

    print("calculate the candidate token embedding...")  
    candidate_token_embeddings = []  

    candidate_token_ids = [index for index, _ in enumerate(candidate_tokens)]  
    
    candidate_token_ids = torch.tensor(candidate_token_ids, dtype=torch.long).to(device).unsqueeze(1) # shape = (num_tokens, 1) == (batch_size, seq_len)
    
    # extract the base model from the accelerator, so that we can get embeddings
    # model_unwrapped = accelerator.unwrap_model(model)
    # model_unwrapped.eval() # save resources
    
    
    # 加入判断逻辑
    # candidate_token_embeddings = model_unwrapped.embeddings(candidate_token_ids) # shape = (batch_size, 1, hidden_size) == (num_tokens, 1, hidden_size)
    candidate_token_embeddings = get_vocab_embeddings_from_model(model, candidate_token_ids).to(device)
    candidate_token_embeddings =  candidate_token_embeddings.squeeze(1) # shape = (num_tokens, hidden_size)
    
    print("candidate token embeddings shape: ", candidate_token_embeddings.shape)

    print("find each centroid a most similar token...")  

    cluster_labels = []  
    cluster_label_embeddings = []
    

    for idx, centroid in enumerate(centroids):  
        max_similarity = -1  
        best_token_id = -1  
        #  Calculate the similarity with all candidate tokens
        for token_id, embedding in enumerate(candidate_token_embeddings):  
            # return ndarray(n_samples_X, n_samples_Y)
            similarity:np.ndarray = cosine_similarity([centroid], [embedding.detach().cpu().numpy()])[0][0]  
            if similarity > max_similarity:  
                max_similarity = similarity  
                best_token_id = token_id  
        cluster_labels.append((idx, best_token_id, max_similarity))  
        cluster_label_embeddings.append(candidate_token_embeddings[best_token_id])

 
    for idx, best_token_id, similarity in cluster_labels:  
        print(f"Cluster {idx + 1}'s best suit token：{candidate_tokens[best_token_id]}, similarity: {similarity:.4f}") 
        classes.append(candidate_tokens[best_token_id])
        
    cluster_label_embeddings = torch.concat(cluster_label_embeddings, dim=0) # shape = (num_topics, hidden_size)

    if not os.path.exists(os.path.dirname(final_label_embeddings_path)):
        os.makedirs(os.path.dirname(final_label_embeddings_path))
        
    print(f"Saving final label embeddings to {final_label_embeddings_path}")  
    torch.save(cluster_label_embeddings, final_label_embeddings_path) 
    
    # 保存元信息 
    with open(final_label_metadata_path, 'w') as f:  
            json.dump(final_metadata, f) 
        
    return cluster_label_embeddings
        
def get_classes_by_lda(dataset_path, model, tokenizer, embedding_size, num_topics = 5, K=5, max_length=512)->List[torch.Tensor]:
    '''
    use the LDA model to extract the class labels
    
    num_topics: number of classes to be generated == num_suffix_tokens
        
    K : number of sub-classes to be generated
    '''
    print("get class labels by latent Drichtlet Allocation (LDA) Model")
    classes = []
    model = model.eval()
    device = Config['device']
    model.to(device)
    
    all_pooled_embeddings = [] # store the pooled embeddings of every 1000 examples
    
    vocab_size = tokenizer.vocab_size
    # count the word frequency of each word in vocab on the training set
    word_freq = {token_id:0 for token_id in range(vocab_size)}
    
    
    train_ds = reformat_input(dataset_path, tokenizer, max_length=max_length)
    
    train_data_loader = DataLoader(train_ds, batch_size=1000, 
                                   collate_fn=default_data_collator, 
                                   num_workers=NUM_CPU_PROCESSES, # use as your need
                                   shuffle=False)
    
    
    processed_questions: List[List[str]] = reformat_input(dataset_path, tokenizer, max_length=max_length, reformat_type = 'lda')

    print("create dictionary and corpus...")  
    # 创建词典：词语到id的映射  
    dictionary = corpora.Dictionary(processed_questions)  
    # 过滤极端词汇（可选）  
    dictionary.filter_extremes(no_below=2, no_above=0.5)  
    # 将文档转换为词袋(Bag-of-Words)表示  
    corpus: List[List[tuple]] = [dictionary.doc2bow(text) for text in processed_questions]  
        
    
    # train LDA model  
    print(f"Training LDA model, class label number = {K}...")  
    lda_model = models.ldamodel.LdaModel(  
        corpus=corpus,  
        id2word=dictionary,  
        num_topics=num_topics,  
        random_state=42,  
        passes=10,  # training epochs
        iterations=50  
    )  
  
    print("Extract the class labels...")  
    for idx, topic in lda_model.show_topics(formatted=False, num_words=K, num_topics=num_topics):  
        topic_words = [word for word, prob in topic]  
        # print(", ".join(topic_words))  
        classes.append(topic_words[0])
    
    print("class labels = ", classes)
    
    
    print("transfer class labels to label embeddings...")  
    topic_word_embeddings = []

    with torch.no_grad():  
        for id, word in enumerate(classes):  
        
            encoded_input = tokenizer(
                word, 
                padding = "max_length",
                truncation=True,  
                max_length=5,  # we set that each label can be divided into 5 tokens
                add_special_tokens=True,
                return_tensors='pt').to(device)  
            
            model_output = model(**encoded_input)  
            # 使用 [CLS] 标记的向量作为词嵌入  
            word_embedding = model_output.last_hidden_state[:, 0, :].squeeze(0)  # [hidden_size]  
            topic_word_embeddings.append(word_embedding)  


        
    return topic_word_embeddings




def train_baas_prompt(config:BaasPromptConfig, chain_encode_args:ChainEncodingArguments):
    setup_distributed()
    
    fix_seed(config.seed)
    
    model_name = config.model_name
    dataset_name = config.dataset_name
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

    
    
    

    
    # the preprocessed dataset only contains ["input_ids", "attention_mask", "labels"]
    
    processed_ds = preprocess_dataset_peft(dataset_name, model_path, max_length = max_length, seq_cls_type=config.seq_cls_type)
    
    
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
            # shuffle=True, # shuffle is not necessary when using DistributedSampler
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            # pin_memory=True,
            sampler=train_sampler
        )
    
    eval_dataloader = DataLoader(
            eval_ds, 
            collate_fn=default_data_collator, 
            batch_size=batch_size,
            # pin_memory=True,
            sampler=eval_sampler
        )
    
    exp_name = f"{config.peft_method}_{model_name}_{dataset_name}"
    tracker = SwanLabTracker("BAAS_PROMPT_TRAING", experiment_name=exp_name)  # 训练可视化
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,  
        # mixed_precision=config.mixed_precision,
        log_with=[tracker]
    )
    tracker_config = {
        "num_epoch": config.num_epochs,
        "batch_num": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }
    accelerator.init_trackers("BAAS_PROMPT_TRAING", config=tracker_config)
    
    baas_model = BassPromptModel(  
        model=model,
        tokenizer=tokenizer,
        config=config,
        chain_encode_args=chain_encode_args,
        device = accelerator.device,
        debug=config.debug
    )
    
    baas_model.to(accelerator.device)
 
    
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
        logger_name = f"{model_name}_{config.peft_method}_{dataset_name}_{config.seq_cls_type}"
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)  
            print(f"已创建新的log存储路径: {logging_dir}") 
        logger = get_logger(name=logger_name, logging_dir=logging_dir, log_level="INFO")

    if accelerator.is_main_process:  
        accelerator.init_trackers("training") 
        
    global_step = 0
    optimizer.zero_grad()
    
    print("\n\n**************************************************************************************")
    print(f"************* Start training model {model_name} on {dataset_name} using {config.peft_method} ... ************\n\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if train_sampler is not None:  # 每轮训练都打乱顺序
            train_sampler.set_epoch(epoch) 
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
                
                loss:torch.Tensor = criterion(logits, labels.long())
                total_loss += loss.detach().float().item() 
                

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
            
            
        if accelerator.is_local_main_process:
            print(f"begin epoch {epoch} evaluating...")    
            
            # if step == len(train_dataloader)-1:  
        eval_results = evaluate_baas_prompt(model, accelerator, eval_dataloader)
        accelerator.wait_for_everyone()  

                
                
                # # 计算评价指标  
                # accuracy = np.mean(np.array(all_preds) == np.array(all_labels))  
                # precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')  

                # accelerator.wait_for_everyone() 

                # print(f"Step {global_step}, Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")  
                
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
    model_path = get_model_path_by_model_name(model_name)

    dataset_path = Config["datasets"][dataset_name]
    model, tokenizer = prepare_model_tokenizer(model_path, AutoModelForSequenceClassification)
    
    max_seq_length = get_max_length_from_model(model)
    import math
    prefix_length = 10
    suffix_length = math.floor(0.1 * max_seq_length)
    
    hidden_size = get_hidden_size_using_model(model)

    config = BaasPromptConfig(
        model_name = model_name,
        model_path = model_path,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        num_epochs=5,
        num_labels=4,
        all_layers=False,
        is_prefix=False,
        prefix_projection=True,
        prefix_hidden_size=hidden_size,
        encoder_hidden_size=hidden_size,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        batch_size=4,
        debug=False,
        seq_cls_type="multiple"
        
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
    
    
    