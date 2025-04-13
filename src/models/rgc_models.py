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
from sentence_transformers import SentenceTransformer, models
from dataclasses import dataclass
from collections import Counter


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
    classes_initiate_method:str = 'cluster' # ['normal','lda','cluster']
    train_size:int = 22000
    mixed_precision:bool=False
    suffix_ratio:int=10


class BaasPromptEncoder(nn.Module):  
    def __init__(  
        self,   
        config: BaasPromptConfig,
        model_config: AutoConfig,
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
        debug:bool=False,
        num_options=None,
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
        self.classes_initiate_method = config.classes_initiate_method
        self.num_options=num_options
        
        
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
            classes_initiate_method = self.classes_initiate_method,
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
            if self.num_options:
                position_ids = torch.arange(  
                    0, sequence_length,   
                    dtype=torch.long,   
                    device=self.device  
                ).expand(self.config.batch_size*self.num_options, -1)  # [batch_size, seq_length] 
            else:
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
            
            if self.num_options:
                assert batch_size == self.config.batch_size*self.num_options, "input embedding's batch_size must be equal to config.batch_size*self.num_options"
            else:
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
            
            
            if self.num_options:
                prefix_embeds=prefix_embeds.repeat(self.num_options, 1, 1)
                suffix_embeds=suffix_embeds.repeat(self.num_options, 1, 1)
            
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
            
            if self.model_type == "bert" or self.model_type == "roberta":
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
            elif self.model_type == "qwen2": # qwen2
                outputs = self.base_model(  
                    # input_ids=input_ids,                 
                    inputs_embeds=inputs_embeds,  # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是从input_ids重新计算
                    attention_mask=attention_mask,  
                    position_ids=position_ids if self.model_type == "bert" or self.model_type == "roberta" else None,
                    # labels=labels,
                    output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                    output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                    return_dict=return_dict,
                    # past_key_values.shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
                    # past_key_values=past_key_values if config.all_layers else None, 
                )   
            elif self.model_type == "gpt2": # qwen2
                outputs = self.base_model(  
                    # input_ids=input_ids,                 
                    inputs_embeds=inputs_embeds,  # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是从input_ids重新计算
                    attention_mask=attention_mask,  
                    position_ids=position_ids if self.model_type == "bert" or self.model_type == "roberta" else None,
                    # labels=labels,
                    output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                    output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                    return_dict=return_dict,
                    # past_key_values.shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
                    # past_key_values=past_key_values if config.all_layers else None, 
                )   
            else: # 
                outputs = self.base_model(  
                    # input_ids=input_ids,                 
                    inputs_embeds=inputs_embeds,  # 直接使用已经计算好的词嵌入（word embeddings，比如从bert提取的）, 而不是从input_ids重新计算
                    attention_mask=attention_mask,  
                    position_ids=position_ids if self.model_type == "bert" or self.model_type == "roberta" else None,
                    # labels=labels,
                    output_attentions=output_attentions, # 当设置为 True 时，模型会在输出中包含每一层的注意力权重（attention weights）
                    output_hidden_states=output_hidden_states, # 当设置为 True 时，模型会在输出中包含每一层的隐藏状态
                    return_dict=return_dict,
                    # past_key_values.shape = (2*n_layer, batch_size, n_head, prefix_length, hidden_size // n_head)
                    # past_key_values=past_key_values if config.all_layers else None, 
                )   
            
            # pooled_output = outputs[1] # shape = (batch_size, hidden_size)
            # cls_token = outputs.hidden_states[-1][:, 0, :] # shape = (batch_size, hidden_size)
            # cls_token = outputs.last_hidden_state[:, 0, :] # shape = (batch_size, hidden_size)

            
            if self.model_type == 'qwen2' or self.model_type == 'gpt2':
                # 使用最后一个非padding token的隐藏状态  
                last_hidden_state = outputs.last_hidden_state  
                # sequence_lengths = attention_mask.sum(dim=1) - 1  
                # sequence_length
                batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0] 
                
                # sequence_output = last_hidden_state[  
                #     torch.arange(batch_size, device=self.device),   
                #         sequence_lengths  
                #     ]  # shape = (batch_size, hidden_size)
                
                # logits= self.classifier(sequence_output)
                logits= self.classifier(last_hidden_state) # shape = (batch_size, seq_len, num_labels)
                
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
                
                # 获取每个序列最后一个非填充token的logits  
                logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] # shape = (batch_size, num_labels)
                
            
            else:
                if hasattr(self.classifier, 'dense') or hasattr(self.classifier, 'out_proj'):
                    # it means that the classifier is a RobertaClassificationHead
                    logits:torch.Tensor = self.classifier(outputs.last_hidden_state) # shape = (batch_size, num_labels)
                else:
                    # last_hidden_state[:,0,:].shape = (batch_size, hidden_size)
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
        
        classes_initiate_method:
            "normal": use the tf-iqv-as to initialize the prefix tokens
            "cluster": use the cluster method to initialize the prefix tokens
            "lda": use the lda method to initialize the prefix tokens
        
        
        return tensor shape = (num_prefix_tokens, embedding_size)
        """
        # model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
        
        class_embeddings = None
        if classes_initiate_method == "normal":
            class_embeddings = get_classes_for_dataset(dataset_path, model, tokenizer, config=config, embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
        elif classes_initiate_method == "cluster":
            class_embeddings:torch.Tensor = get_classes_by_clustering(dataset_path, model, tokenizer, config=config ,embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
        elif classes_initiate_method == "lda":
            class_embeddings = get_classes_by_lda(dataset_path, model, tokenizer, config=config, embedding_size = hidden_size, num_topics=num_topics, max_length=max_length)
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