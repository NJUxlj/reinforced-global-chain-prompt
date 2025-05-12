import torch
import torch.nn as nn
import torch.nn.functional as F  

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    PretrainedConfig
)
from accelerate import Accelerator
from config.config import Config, BudgetSchedulerConfig   

from typing import List, Tuple, Dict, Optional, Any, Union
import math


from sentence_transformers import SentenceTransformer, models

from src.utils.wrapper import *
from config.config import *
from src.utils.utils import (
    get_base_model_using_model,
)

'''

this file contains various model components, such as:

prompt encoder, cross-attention, gated-cross-attention, sparse-attention ...


This file defines all the model components that are required during the RGC-Tuning


'''


class InputEncoder(nn.Module):
    '''
    function:
        1. compress the length of the  real input sequence into [max_seq_length - prefix_length - suffix_length - hard_prompt_length] 
        2. 使用滑动窗口+自适应池化+注意力加权来将MCQ的input压缩到[max_seq_length - prefix_length - suffix_length - hard_prompt_length]
        3. 保留重要信息的同时满足max_length限制。
        
    Args:  
        target_length (int): 目标序列长度  
        hidden_size (int): 隐藏层维度  
        window_size (int): 滑动窗口大小，默认128  
        stride (int): 滑动步长，默认64  
        pooling_mode (str): 池化模式，可选'max'或'avg'，默认'max'  
    '''
    def __init__(  
        self,   
        target_length: int,  
        hidden_size: int,  
        window_size: int = 128,  
        stride: int = 64,  
        pooling_mode: str = 'max'  
    ):  
        super().__init__()
    
        self.target_length = target_length  
        self.hidden_size = hidden_size  
        self.window_size = window_size  
        self.stride = stride  
        self.pooling_mode = pooling_mode  
        
        # 注意力层用于给不同窗口分配权重  
        self.attention = nn.Sequential(  
            nn.Linear(hidden_size, hidden_size // 2),  
            nn.ReLU(),  
            nn.Linear(hidden_size // 2, 1)  
        )  

        # 用于将压缩后的序列映射到目标长度  
        '''
        平均池化层
            它的特点是：可以将任意长度的输入序列压缩或扩展到指定的目标长度。
            这个操作是自适应的，意味着它会自动调整池化窗口的大小，以确保输出恰好是目标长度。
        
        '''
        self.length_adapter = nn.AdaptiveAvgPool1d(target_length)
    
    def get_windows(  
        self,   
        sequence: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None  
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        '''  
        使用滑动窗口切分序列  
        
        Args:  
            sequence: shape (batch_size, seq_len, hidden_size)  
            attention_mask: shape (batch_size, seq_len)  
            
        Returns:  
            windows: shape (batch_size, num_windows, window_size, hidden_size)  
            window_masks: shape (batch_size, num_windows, window_size)  
        ''' 
         
        batch_size, seq_len, hidden_size = sequence.shape  
        
        # 计算窗口数量  
        num_windows = max(1, (seq_len - self.window_size) // self.stride + 1)  # 如果最后一个窗口超出max_length边界，直接舍弃
        
        windows = []  
        window_masks = []  
        
        for i in range(num_windows):  
            start_idx = i * self.stride  
            end_idx = start_idx + self.window_size  
            
            # 获取当前窗口  
            window = sequence[:, start_idx:end_idx, :]  # shape = (batch_size, window_size, hidden_size)
            windows.append(window)  
            
            if attention_mask is not None:  
                window_mask = attention_mask[:, start_idx:end_idx]  
                window_masks.append(window_mask)  
        
        windows = torch.stack(windows, dim=1)  # (batch_size, num_windows, window_size, hidden_size)  
        
        if attention_mask is not None:  
            window_masks = torch.stack(window_masks, dim=1)  # (batch_size, num_windows, window_size)  
        else:  
            window_masks = torch.ones_like(windows[..., 0])   #  # windows[..., 0] 取最后一维的第一个元素，保持前面的维度不变
            
        return windows, window_masks  
    
    def pool_windows(  
        self,   
        windows: torch.Tensor,  
        window_masks: torch.Tensor  
    ) -> torch.Tensor:  
        '''  
        对每个窗口进行池化操作  
        
        Args:  
            windows: shape (batch_size, num_windows, window_size, hidden_size)  
            window_masks: shape (batch_size, num_windows, window_size)  
                    mask矩阵，1表示有效token，0表示padding
            
        Returns:  
            pooled: shape (batch_size, num_windows, hidden_size)  
        '''  
        batch_size, num_windows, window_size, hidden_size = windows.shape  
        
        # 展平batch和window维度  
        flat_windows = windows.view(-1, window_size, hidden_size)  # (batch_size * num_windows, window_size, hidden_size)  
        flat_masks = window_masks.view(-1, window_size)  # (batch_size * num_windows, window_size)  
        
        # 应用mask, 将无效token（mask=0）的特征置为0  
        # flat_masks.unsqueeze(-1).shape = (batch_size * num_windows, window_size, 1)
        masked_windows = flat_windows * flat_masks.unsqueeze(-1)   # shape = (batch_size * num_windows, window_size, hidden_size)
        
        if self.pooling_mode == 'max':  # 最大池化的情况 
            # masked_fill 将mask为1的位置（原始mask为0的位置）填充为负无穷  
            masked_windows = masked_windows.masked_fill(
                    ~flat_masks.unsqueeze(-1).bool(), 
                    float('-inf')
                ) # ~flat_masks 进行取反操作：0变1，1变0  
            
            pooled = torch.max(masked_windows, dim=1)[0]  # shape = (batch_size * num_windows, hidden_size)
            
            '''
            torch.max(input, dim) 会返回一个元组 (tuple)，包含两个元素：
                [0]: 最大值 (values)
                [1]: 最大值的索引位置 (indices)
            '''
        else:  # avg pooling  
            # 计算每个窗口的有效token数量  
            valid_tokens = flat_masks.sum(dim=1, keepdim=True)  # (batch_size * num_windows, 1)  
            pooled = masked_windows.sum(dim=1) / (valid_tokens + 1e-10)  # (batch_size * num_windows, hidden_size)  
        
        return pooled.view(batch_size, num_windows, hidden_size) 
    
    def forward(  
        self,   
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None  
    ) -> torch.Tensor:  
        '''  
        前向传播函数  
        
        Args:  
            hidden_states: shape (batch_size, seq_len, hidden_size)  
            attention_mask: shape (batch_size, seq_len)  
            
        Returns:  
            compressed_sequence: shape (batch_size, target_length, hidden_size)  
        '''  
        # 1. 获取滑动窗口  
        windows, window_masks = self.get_windows(hidden_states, attention_mask)  
        
        # 2. 对每个窗口进行池化  
        pooled_windows = self.pool_windows(windows, window_masks)  # (batch_size, num_windows, hidden_size)  
        
        # 3. 计算注意力权重  [每个窗口分配一个注意力分数]
        attention_scores = self.attention(pooled_windows)  # (batch_size, num_windows, 1)  
        attention_weights = F.softmax(attention_scores, dim=1)  
        
        # 4. 加权求和  
        weighted_windows = pooled_windows * attention_weights  # (batch_size, num_windows, hidden_size)
        
        # 5. 使用自适应池化调整序列长度  
        # 转换维度以适应AdaptiveAvgPool1d  
        sequence = weighted_windows.transpose(1, 2)  # (batch_size, hidden_size, num_windows)  
        compressed = self.length_adapter(sequence)  # (batch_size, hidden_size, target_length)  
        compressed = compressed.transpose(1, 2)  # (batch_size, target_length, hidden_size)  
        
        return compressed  


class TextDataset(Dataset):  
    """
    用于将文本转换为Dataset格式，便于并行处理
    
    用于处理SentenceEncoder编码的文本列表，先将其转为TextDataset, 再转为DataLoader, 最后放入Accelerator
    """  
    def __init__(self, texts):  
        if isinstance(texts, str):  
            self.texts = [texts]  
        else:  
            self.texts = texts  

    def __len__(self):  
        return len(self.texts)  

    def __getitem__(self, idx):  
        if isinstance(idx, int):
            return self.texts[idx]  
        elif isinstance(idx, list):
            return [self.texts[id] for id in idx]
        else:
            raise TypeError(f"Index must be an integer, got {type(idx)}") 
        
        
        

class CustomSentenceEncoder(nn.Module):
    def __init__(self, hidden_size, model):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size) # 用于维度转换
        self.activation = nn.GELU()




        


class SentenceEncoder(nn.Module):
    def __init__(
            self, 
            hidden_size,
            model = None
        ):
        '''
        内含 Sentence Transformer, 区别在于可以将句子编码后的hidden_size自动调整为用户输入的hidden_size
        
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.encoder_name = SENTENCE_TRANSFORMER_PATH
        
        encoder = SentenceTransformer(self.encoder_name)
        sentence_embedding_dimension = encoder.get_sentence_embedding_dimension()
    
        if sentence_embedding_dimension != self.hidden_size and self.hidden_size != 0: # 0 means no dimension adaption
            print(f"the hidden_size of your requirement {self.hidden_size} does not match the hidden_size of the sentence transformer {sentence_embedding_dimension}")
            print("========= adding dimension adaption layer .......")
            
            word_embedding_model = models.Transformer(self.encoder_name)
            
            # 2. 添加池化层  
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            # 3. 添加降维层（例如将768维降至256维）  
            dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),   
                                    out_features=self.hidden_size,   
                                    activation_function=torch.nn.Tanh())  
            
            encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        
        self.encoder = encoder

    def encode(self, sentences:Union[List[str], str])->torch.Tensor:
        '''
        return:
            shape = [num_sentences, hidden_size]  or [hidden_size]
        
        '''
    
        sentence_embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True
        )  # shape = [K, H]
        
        # # 如果输入是单个句子，则去掉批次维度  
        # if isinstance(sentences, str):  
        #     sentence_embeddings = sentence_embeddings.squeeze(0) 
        

        
        return sentence_embeddings
    
    def forward(self):
        pass




class SparseAttention(nn.Module):
    '''
    Adaptive Sparse Attention with Local-Global Information
    '''
    def __init__(self, hidden_size, num_heads=8, local_window=3, sparsity_threshold=0.1): 
        pass
    

    



# class MultiHeadCrossAttention(nn.Module):  
#     def __init__(self, hidden_size, num_heads=8):  
#         super().__init__()  
#         self.num_heads = num_heads  
#         self.head_dim = hidden_size // num_heads  
        
#         self.q_proj = nn.Linear(hidden_size, hidden_size)  
#         self.k_proj = nn.Linear(hidden_size, hidden_size)  
#         self.v_proj = nn.Linear(hidden_size, hidden_size)  
#         self.o_proj = nn.Linear(hidden_size, hidden_size)  
        
#         self.attention_weights = None  # 存储注意力权重用于可视化  
        
#     def forward(self, query, key, value):  
#         B, N, C = query.shape  
        
#         # 多头投影  
#         q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim)  
#         k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim)  
#         v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim)  
        
#         # 注意力计算  
#         attn = torch.einsum('bnhd,bmhd->bnmh', q, k) / math.sqrt(self.head_dim)  
#         attn = F.softmax(attn, dim=2)  
#         self.attention_weights = attn  # 保存注意力权重  
        
#         # 输出  
#         out = torch.einsum('bnmh,bmhd->bnhd', attn, v)  
#         out = out.reshape(B, N, -1)  
#         out = self.o_proj(out)  
        
#         return out  
    
    
    





class BaasPromptV2(torch.nn.Module):  
    def __init__(self, 
                 model, 
                 prefix_embeddings, 
                 suffix_embeddings, 
                 num_prefix_tokens, 
                 num_suffix_tokens,
                 min_step=3,
                 max_step=8,
                 num_labels=4):  
        super(BaasPromptV2, self).__init__() 
        '''
        使用SVD动态调整每一层推理链长度的BaasPrompt
        
        
        
        :param: model: 
            AutoModelForSequenceClassification
            BertForSequenceClassification
            Qwen2ForSequenceClassification
        
        ''' 
        self.model = model  
        self.device = Config['device']
        self.prefix_embeddings = prefix_embeddings  
        self.suffix_embeddings = suffix_embeddings 
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.embedding_layer = self.model.get_input_embeddings()
        
        self.base_model =  get_base_model_using_model(self.model)
        self.config = self.base_model.config 
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size 
        self.min_step = min_step
        self.max_step = max_step
        self.num_labels = num_labels
        
        
        scheduler_config = BudgetSchedulerConfig(
            
        )
        
        # 初始化FixedSVD对象  
        self.fixed_svd = FixedRankSVD(  
            num_layers=self.num_layers,  
            min_step=self.min_step,  
            max_step=self.max_step,  
            hidden_size=self.hidden_size,
            config = scheduler_config,
        )  
        
        # 替换bert模型中的attention模块 , 这是是初始化所有层的prefix和suffix
        for i, layer in enumerate(self.base_model.encoder.layer):  
            layer.attention.self = BaasAttention(  
                config=self.config,  
                layer_idx=i,  
                fixed_svd=self.fixed_svd,
                prefix_embeddings=self.prefix_embeddings,
                suffix_embeddings=self.suffix_embeddings,
            )  
            
            self.fixed_svd.suffix_weights[i] = suffix_embeddings.clone()
        
        # 冻结分类器参数  
        for param in self.base_model.classifier.parameters():  
            param.requires_grad = False  
    
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
            prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=self.device)  
            suffix_mask = torch.ones(batch_size, self.num_suffix_tokens, device=self.device)  

            # print(f"attention_mask.shape = {attention_mask.shape}")
            # print(f"prefix_mask.shape = {prefix_mask.shape}")
            # print(f"suffix_mask.shape = {suffix_mask.shape}")
            
            attention_mask = attention_mask.squeeze(1)
            attention_mask = torch.cat([prefix_mask, attention_mask, suffix_mask], dim=1)  # (4, 522)
            
            # print(f"attention_mask.shape after concat = {attention_mask.shape}") 
        
        if token_type_ids is not None:  
            prefix_type_ids = torch.zeros(batch_size, self.num_prefix_tokens, device=self.device)
            suffix_type_ids = torch.zeros(batch_size, self.num_suffix_tokens, device=self.device)
            
            # print(f"token_type_ids.shape = {token_type_ids.shape}")
            # print(f"prefix_type_ids.shape = {prefix_type_ids.shape}")
            # print(f"suffix_type_ids.shape = {suffix_type_ids.shape}")
            
            token_type_ids = token_type_ids.squeeze(1)
            token_type_ids = torch.cat([prefix_type_ids, token_type_ids, suffix_type_ids], dim=1)
            
            token_type_ids = token_type_ids.long() # (4, 522)
            
            # print(f"token_type_ids.shape after concat = {token_type_ids.shape}")

        
        
        # 每一个batch的反向传播结束以后， 更新所有层的suffix embedding
        
        # 所以应该写在训练循环里面 
        
        # 调用原始模型的forward方法  
        outputs = self.model(  
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids,
            labels=labels  
        )  
        
        return outputs  
    
    def update_all_suffix_embeddings(self, suffix_embeddings):
        '''
         更新模型中所有层的后缀嵌入
        '''
        
        # 此处需要在每个训练步修改  self.fixed_svd 中的参数，
        # 这样动态更新才有效果
        
        
        self.fixed_svd.extend_all_layer_matrices()
        
        suffix_weights:Dict[int, torch.Tensor] = {}
        
        for i in range(self.num_layers):
            P = self.fixed_svd.P_matrices[i]
            Lambda = self.fixed_svd.Lambda_matrices[i]
            Q = self.fixed_svd.Q_matrices[i]
            
            suffix_embedding = torch.mm(torch.mm(P,Lambda),Q)
        
            suffix_weights[i] = suffix_embedding
        
        # 将suffix_weights更新到Bert中的每一层
        for i, layer in enumerate(self.base_model.encoder.layer):  
            layer.attention.self = BaasAttention(  
                config=self.config,  
                layer_idx=i,  
                fixed_svd=self.fixed_svd,
                prefix_embeddings=self.prefix_embeddings,
                suffix_embeddings=suffix_weights[i],
            )  
        

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



class BaasAttention(nn.Module):  
    """自定义的attention模块，用于插入suffix embeddings"""  
    def __init__(
        self, 
        config:PretrainedConfig, 
        layer_idx, 
        fixed_svd:'FixedRankSVD'=None, 
        prefix_embeddings = None, 
        suffix_embeddings = None,
        is_prefix = False
        ):  
        '''
        :param config: base_model 的配置文件
        :param layer_idx: base_model的隐藏层id
        :param fixed_svd: FixedSVD对象，用于动态调整每一层的rank(推理步数，即，suffix_length)
        :param prefix_embeddings: 前缀嵌入, shape=(batch_size, prefix_length, hidden_size), 但是batch维不一定有
        :param suffix_embeddings: 后缀嵌入, shape=(batch_size, suffix_length, hidden_size), 但是batch维不一定有

        :param is_prefix: 是否把prefix embedding 和 suffix embeddings 作为额外的前缀后缀附加到input上
                            True: 附加到input上， 输入的总长度变为 max_length + prefix_length + suffix_length
                            False:类似prompt-tuning 的插入, 输入的总长度还是max_length
        '''
        super().__init__()  
        self.layer_idx = layer_idx  
        self.fixed_svd = fixed_svd  
        self.is_prefix = is_prefix
        
        self.max_length = config.max_position_embeddings
        self.prefix_length = prefix_embeddings.shape[0] if prefix_embeddings.dim()==2 else prefix_embeddings.shape[1]
        self.suffix_length = suffix_embeddings.shape[0] if suffix_embeddings.dim()==2 else suffix_embeddings.shape[1]
        self.input_length = self.max_length - (self.prefix_length + self.suffix_length)
        
        self.prefix_embeddings:torch.Tensor = prefix_embeddings
        self.suffix_embeddings:torch.Tensor = suffix_embeddings
        
        # 多头注意力的配置  
        self.num_attention_heads = config.num_attention_heads  
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  
        self.all_head_size = self.num_attention_heads * self.attention_head_size  
        
        # 原始的Q,K,V投影矩阵  W_q, W_k, W_v
        self.query = nn.Linear(config.hidden_size, self.all_head_size)  
        self.key = nn.Linear(config.hidden_size, self.all_head_size)  
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  
        
        # dropout和输出投影  
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  
        
    def transpose_for_scores(self, x:torch.Tensor):  
        """将张量转换为多头格式"""  
        batch_size, seq_len, _ = x.size()  
        x = x.view(batch_size, seq_len, self.num_attention_heads,   
                  self.attention_head_size)  
        return x.permute(0, 2, 1, 3)  # shape = (batch_size, num_heads, seq_len, head_size)
        
    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        head_mask=None, 
        output_attentions=False
        ):  
        '''
        :param hidden_states: shape=(batch_size, input_length, hidden_size)
        
        '''
        batch_size, seq_length, _ = hidden_states.size()  
        
        # 1. 获取当前层的suffix embeddings  
        # P, Lambda, Q = self.fixed_svd.get_layer_matrices(self.layer_idx)  
        # suffix_embeddings = torch.matmul(torch.matmul(P, Lambda), Q)  
        
        suffix_embeddings = None 
        prefix_embeddings = None
        
        # 检查并扩展维度
        if self.suffix_embeddings.dim()==3:
            suffix_embeddings = self.suffix_embeddings.repeat(batch_size, 1, 1)
        else:
            suffix_embeddings = self.suffix_embeddings.unsqueeze(0).expand(  
                batch_size, -1, -1)  
        
        if self.prefix_embeddings.dim()==3:
            prefix_embeddings = self.prefix_embeddings.repeat(batch_size, 1, 1)
        else:
            prefix_embeddings = self.prefix_embeddings.unsqueeze(0).expand(  
                batch_size, -1, -1)  
        
        # 将原始输入和suffix embeddings拼接  
        truncated_hidden_states = hidden_states[:]
        if not self.is_prefix: # p-tuning style
            truncated_hidden_states = truncated_hidden_states[:, :self.input_length, :]
            
        extended_hidden_states = torch.cat(  
            [prefix_embeddings, truncated_hidden_states, suffix_embeddings], dim=1)  
        
        # 4. 计算Q,K,V  
        query_layer = self.query(hidden_states)  # shape = (hidden_size, hidden_size) * (batch_size, seq_len, hidden_size) = (batch_size, seq_len, hidden_size)
        key_layer = self.key(extended_hidden_states)  # shape = (batch_size, seq_len_extended, hidden_size)
        value_layer = self.value(extended_hidden_states)  # shape = (batch_size, seq_len_extended, hidden_size)
        
        # 5. 转换为多头格式  
        query_layer = self.transpose_for_scores(query_layer)  # # shape = (batch_size, num_heads, seq_len, head_size)
        key_layer = self.transpose_for_scores(key_layer)   # shape = (batch_size, num_heads, seq_len_extended, head_size)
        value_layer = self.transpose_for_scores(value_layer)  
        
        # 6. 计算注意力分数  
        # 交换最后两个维度
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   # shape = (batch_size, num_heads, seq_len, seq_len_extended)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  
        
        
        '''
        
        在transformer的注意力计算中，attention mask的完整维度格式是:
        [batch_size, num_attention_heads, seq_length, seq_length]

        但为了节省内存和计算效率，我们通常使用更紧凑的格式:
        [batch_size, 1, 1, seq_length]

        这里的两个1允许通过广播机制自动扩展到完整维度
        
        第一个seq_length：

            查询(query)序列的长度
            表示"我要查询谁"的位置
            
        第二个seq_length：

            键(key)序列的长度
            表示"我可以被查询的"位置
        
        '''
        
        
        # 7. 处理attention mask  
        if attention_mask is not None:  
            
            if not self.is_prefix:
                attention_mask = attention_mask[:,:self.input_length]
            
            attention_mask= attention_mask.unsqueeze(1).unsqueeze(1)  # shape = (batch_size, 1, 1,seq_length)
            # 扩展attention mask以包含suffix tokens  
            suffix_mask = torch.ones(  
                (batch_size, 1, 1, suffix_embeddings.size(1)),   
                device=attention_mask.device  
            )  
            
            prefix_mask = torch.ones(  
                (batch_size, 1, 1, prefix_embeddings.size(1)),   
                device=attention_mask.device  
            ) 
            extended_attention_mask = torch.cat(  
                [prefix_mask, attention_mask, suffix_mask], dim=-1)  # shape = (batch_size, 1, 1, seq_len_extended)
            
            attention_scores = attention_scores + extended_attention_mask   # 广播一下  (batch_size, num_heads, seq_len, seq_len_extended)
        
        # 应用softmax和dropout  
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  
        attention_probs = self.dropout(attention_probs)  
        
        # 应用head mask(如果有)  
        if head_mask is not None:  
            attention_probs = attention_probs * head_mask  
        
        # 计算上下文层  
        context_layer = torch.matmul(attention_probs, value_layer)  # shape = (batch_size, num_heads, seq_len, seq_len_extended) * (batch_size, num_heads, seq_len_extended, head_size) = (batch_size, num_heads, seq_len, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # shape = (batch_size, seq_len, num_heads, head_size)
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)  
        
        # 通过输出投影和层归一化  
        output = self.dense(context_layer)  
        output = self.dropout(output)  
        output = self.LayerNorm(output + hidden_states)  # residual
        
        outputs = (output,)  
        if output_attentions:  
            outputs = outputs + (attention_probs,)  
            
        return outputs  



# 使用示例  
def main():  
    # 初始化模型  
    model = BaasPromptV2(  
        model_name='bert-large-uncased',  
        num_prefix_tokens=4,  
        min_step=3,  
        max_step=8,  
        num_labels=4  # RACE数据集的4个选项  
    )  
    
    # 训练配置  
    training_args = TrainingArguments(  
        output_dir="./results",  
        num_train_epochs=3,  
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,  
        warmup_steps=500,  
        weight_decay=0.01,  
        logging_dir="./logs",  
    )  
    
    # 自定义训练器  
    class BaasTrainer(Trainer):  
        def training_step(self, model, inputs):  
            # 每个epoch开始前更新矩阵  
            model.fixed_svd.decompose_all_layer_weights()  
            model.fixed_svd.extend_all_layer_matrices(model.num_layers)  
            
            # 调用父类的training_step  
            return super().training_step(model, inputs)  
    
    # 创建训练器  
    trainer = BaasTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,  
    )  
    
    # 开始训练  
    trainer.train() 


# def compute_satisfaction_score_v3(k, i, P, Q, λ, l_q, L, d1, d2, alpha):  
#     """  
#     方案2: Distance-Aware Gradient Impact Score  
    
#     额外参数:  
#     α: 层深度影响因子  
#     """  
#     # 层深度指数衰减  
#     depth_factor = math.exp(alpha * k / L)  
    
#     # 位置重要性权重（考虑与输入的距离）  
#     position_weight = 1 / (1 + math.exp(-(k - L/2)))  # Sigmoid函数  
    
#     # 问题长度归一化因子  
#     length_norm = math.sqrt(l_q / 30)  # 30是基准长度  
    
#     # 梯度敏感度计算（带距离衰减）  
#     s_lambda = abs(λ[i] * λ[i].grad) if λ[i].grad is not None else 0  
    
#     # P矩阵的梯度计算（考虑行位置的重要性）  
#     if P[:, i].grad is not None:  
#         p_grads = abs(P[:, i] * P[:, i].grad)  
#         position_weights = torch.linspace(1, 0.5, d1)  # 线性衰减权重  
#         s_P = torch.sum(p_grads * position_weights) / d1  
#     else:  
#         s_P = 0  
    
#     # Q矩阵的梯度计算  
#     s_Q = torch.sum(abs(Q[i, :] * Q[i, :].grad)) / d2 if Q[i, :].grad is not None else 0  
    
#     # 组合梯度影响  
#     gradient_impact = (s_lambda + s_P + s_Q) / 3  
    
#     # 最终satisfaction score  
#     score = depth_factor * position_weight * length_norm * gradient_impact  
    
#     return score  





class FixedRankSVD:  
    
    '''
    实现每一层的suffix embedding的分解， 
    每一层的后缀矩阵分解为 P_k, Lambda_k, Q_k
    然后分别扩展为 P_k, Lambda_k, Q_k （使用Sparse Attention)
    
    
    P.shape = (n_step, n_step)
    Lambda.shape = (n_step, n_step)
    Q.shape = (n_step, hidden_size)  注意：Q是已经经过转置的V矩阵
    '''
    def __init__(self, num_layers, hidden_size, min_steps, max_steps, config:BudgetSchedulerConfig):  
        self.config = config
        self.n_layers = config.n_layers  
        self.hidden_size = hidden_size  
        self.min_steps = config.min_rank  # min(n_step)  
        self.max_steps = config.max_rank  # input_seq_length  
        
        
        self.P_matrices = {}  # 存储每层的P矩阵  
        self.Q_matrices = {}  # 存储每层的Q矩阵  
        self.Lambda_matrices = {}  # 存储每层的Lambda矩阵  
        
        
        self.suffix_weights:Dict[int, torch.Tensor] = {}  # 存储每层的后缀矩阵
        
        
        # self.momentum = {}  
        # self.velocity = {}  
        
        self.satisfaction_scores = {}
        
        self.estimator = AdaptiveSingularValueEstimator(
            config = config
        )
    
    

        
    def decompose_layer_weights(self, layer_id):  
        """  
        对指定层的权重进行SVD分解  
        
        Args:  
            layer_id (int): 层的索引  
            
        Returns:  
            tuple: (P, Lambda, Q) 分解后的三个矩阵  
        """  
        # 1. 获取该层的suffix权重矩阵  
        if layer_id not in self.suffix_weights:  
            raise ValueError(f"Layer {layer_id}'s suffix weights not found")  
        
        weight_matrix = self.suffix_weights[layer_id]  # shape = (min_step, hidden_size)
        
        # 2. 使用estimator获取该层应该使用的rank数  
        target_rank = self.min_steps
         
        
        # 3. 执行SVD分解  
        try:  
            U, S, V = torch.svd(weight_matrix)  
        except Exception as e:  
            print(f"SVD decomposition failed for layer {layer_id}: {e}")  
            # 如果常规SVD失败，尝试使用更稳定的方法  
            U, S, V = torch.svd(weight_matrix + 1e-10 * torch.randn_like(weight_matrix))  
        
        # 4. 截断到目标rank  
        P = U[:target_rank, :target_rank]  # [max_steps, target_rank]  
        Lambda = torch.diag(S[:target_rank])  # [target_rank, target_rank]  
        Q = V[:, :target_rank].t  # [hidden_size, target_rank]  
        
        # 5. 存储分解结果  
        self.P_matrices[layer_id] = P  
        self.Lambda_matrices[layer_id] = Lambda  
        self.Q_matrices[layer_id] = Q  
        
        return P, Lambda, Q  
    
    
    def decompose_all_layer_weights(self):
        '''
        对所有suffix层的权重进行分解, 分解到 suffix_weight列表中
        '''
        for i in range(self.n_layers):
            self.decompose_layer_weights(i)
       
    def initiate_layer_weights(self):
        pass
    
    
    def initiate_all_layer_weights(self):
        pass
    
    
    def extend_layer_matrices(self, layer_index, layer_budget):  
        """扩展SVD矩阵的具体实现"""  
        # 扩展P矩阵  
        new_P, new_Lambda = self.compute_new_P_vectors(layer_index, layer_budget, self.estimator)
        
        
        # 扩展Q矩阵  
        new_Q = self.compute_new_Q_vectors(layer_index, layer_budget, self.estimator)
        
        
        # 更新到矩阵列表
        self.P_matrices[layer_index] = new_P
        self.Lambda_matrices[layer_index] = new_Lambda
        self.Q_matrices[layer_index] = new_Q
        
        return new_P, new_Lambda, new_Q  

    
    def extend_all_layer_matrices(self):
        
        '''
        利用预算调度器对所有层的SVD矩阵进行更新
        '''
        total_budget = self.estimator.compute_total_budget(self.Lambda_matrices)
        layer_budgets:Dict[int, int] = self.estimator.allocate_layer_budgets(
            self.Lambda_matrices,
            total_budget
        )
        for i in range(self.n_layers):
            self.extend_layer_matrices(i, layer_budgets[i])
        
        
        # 动态调整推理链长度的本质：
           # 在训练的每一轮将时间步t，以及其他参数传入 AdaptiveSingularValueEstimator
        
        
        
    
    

    def compute_new_P_vectors(self, layer_idx, layer_budget, estimator:"AdaptiveSingularValueEstimator"):  
        """  
        计算新的扩展后的P矩阵，但是首先要获得扩展后的Lambda矩阵 （新的奇异向量必须先算好）
        
        参数:  
        P_k: 形状为[n_step, n_step]的矩阵  

        
        返回:  
        new_P: 形状为[layer_budget, layer_budget]的新P向量矩阵  
        """  
        P = self.P_matrices.get(layer_idx)  
        # Lambda = self.Lambda_matrices.get(layer_idx) 
        
        if P is None:  
            raise ValueError(f"Layer {layer_idx}'s SVD matrices P not initialized")   
        
        current_rank = P.shape[0] 
        
        # 如果目标秩小于等于当前秩，直接截断返回  
        if layer_budget <= current_rank:  
            return P[:layer_budget, :layer_budget]  
        
        # 扩展矩阵  
        new_P = torch.zeros(layer_budget, layer_budget, device=P.device)  
        new_P[:current_rank, :current_rank] = P  
        
        # 扩展Lambda矩阵 , 顺便计算新的奇异值
        new_Lambda = self.compute_new_Lambda_vectors(layer_idx, layer_budget, estimator)
        
        # P矩阵和Lambda矩阵的扩展是同步的, 确保了在生成新的正交向量时能够使用正确的奇异值
        # 逐步添加新的行和列  
        for i in range(current_rank, layer_budget):   
            # 生成新列  
            p_c = self.generate_orthogonal_column(   # 里面使用了Attention， 使用了新的奇异值
                new_P[:i, :i],   
                new_Lambda[:i, :i]  
            )  # shape=[i, 1] 
            
            # 将新列放入扩展矩阵  [扩展为(i-1, i)]
            new_P[:i, i:i+1] = p_c 
            
            # 生成新行  
            P_extended = new_P[:i+1, :i+1]  # 向下延伸一行
            
            p_r = self.generate_orthogonal_row(  
                P_extended,  
                new_Lambda[:i+1, :i+1]  
            )  # shape=[1, i]  

            # 将新行放入扩展矩阵, 列数不变，行数+1， [扩展为(i, i)]
            new_P[i:i+1, :i] = p_r
            
            
            # 计算对角元素以保持正交性  
            row_norm = torch.norm(p_r)  
            col_norm = torch.norm(p_c)  
            p_ii = math.sqrt(1 - row_norm**2 - col_norm**2)  
            new_P[i, i] = p_ii 

        # 更新类中存储的矩阵  
        self.P_matrices[layer_idx] = new_P  
        self.Lambda_matrices[layer_idx] = new_Lambda  
        
        # 在更新完矩阵后，更新satisfaction score  
        self.satisfaction_scores[layer_idx] = self.compute_satisfaction_score(layer_idx) 
        return new_P, new_Lambda
    

    def compute_new_Q_vectors(self, layer_idx, layer_budget, estimator:"AdaptiveSingularValueEstimator"):  
        """扩展Q矩阵到目标秩  
        
        确保在执行此函数之前， P和 Lambda 已经更新完毕
        
        Args:  
            layer_idx: 层索引  
            target_rank: 目标秩（由layer_budget决定）  
            
        Returns:  
            torch.Tensor: 扩展后的Q矩阵  
        """  
        
        Q = self.Q_matrices.get(layer_idx)   # shape = (n_step, h)
        # 扩展Lambda矩阵 , 顺便计算新的奇异值， 如果layer_budgetmei有变，那就返回已有的吗，不会重新计算
        Lambda = self.compute_new_Lambda_vectors(layer_idx, layer_budget, estimator)
        
        if Q is None:  
            raise ValueError(f"Layer {layer_idx}'s SVD Q  matrices not initialized")  
        
        if Lambda is None:  
            raise ValueError(f"Layer {layer_idx}'s SVD Lambda  matrices not initialized")  
        
        current_rank = Q.shape[0]   
        feature_dim = Q.shape[1]  
        
        # 如果目标秩小于等于当前秩，直接截断返回  
        if layer_budget <= current_rank:  
            return Q[:layer_budget, :]  
        
        # 扩展矩阵  
        new_Q = torch.zeros(layer_budget, feature_dim, device=Q.device)  
        new_Q[:current_rank, :] = Q 
        
        # 获取当前的奇异值序列  
        current_singular_values = torch.diag(Lambda)  
        new_Lambda = Lambda.clone()
        
        # 逐步添加新的行  
        for i in range(current_rank, layer_budget):  

            current_singular_values = torch.diag(new_Lambda[:i+1, :i+1])  
            
            
            # # 生成新列  
            # q_c = self.generate_orthogonal_column(  
            #     new_Q[:i, :i],   
            #     new_Lambda[:i+1, :i+1]  # 包含新计算的奇异值  
            # )  # shape=[i, 1]  
            
            # # 将新列放入扩展矩阵  
            # new_Q[:i, i:i+1] = q_c  
            
            # 生成新行  
            Q_extended = new_Q[:i+1, :i] 
            
            # P_extended = self.P_matrices[layer_idx]
            # if len(P_extended) < i:
            #     self.compute_new_P_vectors(layer_idx, layer_budget, estimator)
            #     print(f"Layer {layer_idx}'s SVD P  matrices not initialized for n_step = {i}") 
            
             
            q_r = self.generate_orthogonal_row(   # (i+1, i)*(i, i+1)*(i+1, i) = (i+1, i)
                                                    #  ..... = (n+1, h)
                Q_extended,  # shape = (i+1, i) = (n+1, h)
                new_Lambda[:i+1, :i+1]  # 包含新计算的奇异值  
            )  # shape=[1, i]  
            
            # 将新行放入扩展矩阵  
            new_Q[i:i+1, :i] = q_r  
            
            # # 计算对角元素以保持正交性  
            # row_norm = torch.norm(q_r)  
            # col_norm = torch.norm(q_c)  
            # q_ii = math.sqrt(1 - row_norm**2 - col_norm**2)  
            # new_Q[i, i] = q_ii  
            
        
        # 更新类中存储的矩阵  
        self.Q_matrices[layer_idx] = new_Q  
        self.Lambda_matrices[layer_idx] = Lambda
        
        # 更新satisfaction score  
        # self.satisfaction_scores[layer_idx] = self.compute_satisfaction_score(layer_idx)  
            
        return new_Q  

    
    def compute_new_Lambda_vectors(self, layer_idx, layer_budget, estimator:"AdaptiveSingularValueEstimator"):  
        """计算新的Lambda矩阵向量  

            
            作用：
                将n_step x n_step的lambda矩阵扩展成 layer_budget x layer_budget
                
                如果 layer_budget < n_step, 则直接截断
        Args:  
            layer_idx: 层索引  
            layer_budget: 该层的目标秩  
            
        Returns:  
            torch.Tensor: 扩展后的Lambda矩阵  
        """  
        
        # 再更新所有矩阵之前，先计算满意度分数
        if not self.satisfaction_scores[layer_idx]:
            self.satisfaction_scores[layer_idx] = self.compute_satisfaction_score(layer_idx)
        
        
        
        Lambda = self.Lambda_matrices.get(layer_idx)  
        
        if Lambda is None:  
            raise ValueError(f"Layer {layer_idx} Lambda matrix not initialized")  

        current_rank = Lambda.shape[0]
        
        # 如果目标秩小于等于当前秩，直接截断返回  
        if layer_budget <= current_rank:  
            return Lambda[:layer_budget, :layer_budget]  
        
        
        # 创建新的Lambda矩阵  
        new_Lambda = torch.zeros(layer_budget, layer_budget, device=Lambda.device)  
        new_Lambda[:current_rank, :current_rank] = Lambda 
        
        # 填满多出来的这些奇异值，需要用到另一个类
        
        new_Lambda = estimator.update_layer_singular_values(new_Lambda, layer_idx, layer_budget, self.satisfaction_scores[layer_idx])
        
        
        
        return new_Lambda
    
    def orthogonalize(self, attention_output, Q_k):  
        """  
        对注意力输出进行正交化处理，确保新生成的向量与现有向量正交  
        
        参数:  
        attention_output: 形状为[d]的向量，表示注意力机制生成的新向量  
        Q_k: 形状为[n, d]的矩阵，包含现有的正交基向量  
        
        返回:  
        orthogonalized_vector: 与Q_k中所有向量正交的新向量  
        """  
        # 确保输入是二维张量  
        if attention_output.dim() == 1:  
            attention_output = attention_output.unsqueeze(0)  
        
        # 使用修正的Gram-Schmidt正交化过程  
        v = attention_output.clone()  
        
        for i in range(Q_k.size(0)):  
            q = Q_k[i:i+1]  # 获取第i个基向量  
            # 计算投影  
            proj = torch.mm(torch.mm(v, q.t()), q) / (torch.norm(q) ** 2 + 1e-8)  
            # 从v中减去投影分量  
            v = v - proj  
        
        # 归一化  
        norm = torch.norm(v)  
        if norm > 1e-8:  # 避免除以零  
            v = v / norm  
        else:  
            # 如果向量接近零向量，生成一个随机正交向量  
            v = torch.randn_like(v)  
            # 递归调用直到得到有效的正交向量  
            v = self.orthogonalize(v, Q_k)  
        
        return v.squeeze()  # 返回一维向量
    
    def generate_orthogonal_column(self, P:torch.Tensor, lambda_k:torch.Tensor):  
        """生成正交列向量  [使用sparse attention]
        
        Args:  
            P: 当前P矩阵  shape = (n_step, n_step)
            
        Returns:  
            torch.Tensor: 新的正交列向量  
        """  
        n_step = P.shape[0]  
        
        # Step 1: 计算键值对  
        K = torch.mm(P, P.t())  # [n_step, n_step]
        
        # Step 2: 计算稀疏注意力分数  
        S = torch.mm(K, K.t()) / math.sqrt(n_step)  # d_model = n_step
        
        # Step 3: Top-k 稀疏化  
        k = min(self.config.k, n_step) 
        # values 是一个形状为 [n_step, k] 的矩阵，存储每行的前 k 个最大值；
        # indices 是一个形状为 [n_step, k] 的矩阵，存储这些最大值的索引。 
        values, indices = torch.topk(S, k, dim=-1)  
        S_sparse = torch.zeros_like(S)  
        # 将 values 按照 indices 的位置填充到 S_sparse 中
        S_sparse.scatter_(-1, indices, values)  
        
        # Step 4: 注意力权重  
        A = torch.softmax(S_sparse, dim=-1)  # shape = [n_step, n_step]
        
        # Step 5: 生成列向量  
        c = torch.mm(A, P)  # shape = [n_step, n_step]
        
        # Step 6: 与奇异值相乘，并加入缩放和噪声  
        c = c[:, -1].unsqueeze(1)  # shape=[n_step, 1]  
        lambda_k_last = lambda_k[-1, -1]
        p_c = math.sqrt(lambda_k_last) * c + torch.randn_like(c) * self.config.epsilon  
        
        # Step 7: 正交化 (Gram-Schmidt)  
        for i in range(n_step):  
            p_i = P[:, i].unsqueeze(1)  # shape=[n_step, 1]   
            p_c = p_c - torch.dot(p_c, p_i) / torch.dot(p_i, p_i) * p_i  
            
        # Step 8: 归一化  
        p_c = p_c / torch.norm(p_c)  
        
        return p_c  
    
    def generate_orthogonal_row(self, P_extended, Lambda):  
        """生成正交行向量  
        
        Args:  
            P_extended: 扩展后的P矩阵 [P, p_c]  
            
            P_extended.shape = (n_step+1, n_step+1)
            
            p_c 是刚刚新扩展的列向量
            
            Lambda: 当前奇异值矩阵 shape=[n_step+1, n_step+1]
        Returns:  
            torch.Tensor: 新的正交行向量  
        """  
        # 类似于generate_orthogonal_column的实现  
        # 但是处理扩展后的矩阵  
        n_step = P_extended.shape[0]  
        
        K = torch.mm(P_extended, P_extended.t())  # shape = (n_step+1, n_step+1)
        
        # attention
        S = torch.mm(K, K.t()) / math.sqrt(n_step)  # # shape = (n_step+1, n_step+1)
        
        # top-k for sparse attention
        k = min(self.config.k, n_step)  
        values, indices = torch.topk(S, k, dim=-1)  
        S_sparse = torch.zeros_like(S)  
        S_sparse.scatter_(-1, indices, values)  
        
        
        A = torch.softmax(S_sparse, dim=-1)  
        R = torch.mm(A, P_extended)  # shape = (n_step+1, n_step+1), 如果是Q矩阵，那就是(n+1, h)
        
        #与奇异值相乘并加入噪声  
        # 选择最后一行作为新行向量  
        r = R[-1:, :]  # shape=[1, n_step]  
        lambda_k = Lambda[-1, -1]  # 获取最后一个奇异值  
        p_r = math.sqrt(lambda_k) * r + torch.randn_like(r) * self.config.epsilon 
        
        # 正交化  
        for i in range(n_step):  
            p_i = P_extended[i]  
            p_r = p_r - torch.dot(p_r, p_i) / torch.dot(p_i, p_i) * p_i  
            
        # 归一化  
        p_r = p_r / torch.norm(p_r)  
        
        return p_r  
    
    
    def compute_satisfaction_score(self, layer_idx):  
        """计算每一层的satisfaction score  
        
        satisfaction score基于以下几个指标：  
        1. 重构误差  
        2. 奇异值衰减率  
        3. 正交性保持程度  
        """  
        P = self.P_matrices[layer_idx]  # (n_step, n_step)
        Q = self.Q_matrices[layer_idx]  # (n_step, h)
        Lambda = self.Lambda_matrices[layer_idx]   # (n_step, n_step)
        
        # 1. 计算重构误差 (reconstruction error)  
        X = torch.mm(torch.mm(P, Lambda), Q)  # 重构矩阵  
        X_norm = torch.norm(X, p='fro')  
        recon_error = -torch.log(1e-8 + X_norm)  # 负对数作为惩罚项  
        
        # 2. 计算奇异值衰减率 (singular value decay rate)  
        singular_values = torch.diag(Lambda)  
        decay_rate = singular_values[:-1] / (singular_values[1:] + 1e-8)  
        decay_score = torch.mean(decay_rate)  
        
        # 3. 计算正交性保持程度 (orthogonality preservation)  
        P_ortho = torch.mm(P, P.t())  
        Q_ortho = torch.mm(Q, Q.t())  
        I = torch.eye(P_ortho.shape[0], device=P.device)  
        ortho_error = torch.norm(P_ortho - I, p='fro') + torch.norm(Q_ortho - I, p='fro')  
        
        # 组合所有指标  
        satisfaction = (self.config.alpha_s * (-recon_error) +  # 重构项（负的因为是误差）  
                       self.config.beta_s * decay_score +       # 衰减项  
                       self.config.gamma_s * (-ortho_error))    # 正交项（负的因为是误差）  
        
        return satisfaction  

    def update_satisfaction_scores(self):  
        """更新所有层的satisfaction scores"""  
        scores = []  
        for layer_idx in range(self.config.n_layers):  
            score = self.compute_satisfaction_score(layer_idx)  
            self.satisfaction_scores[layer_idx] = score  
            scores.append(score)  
        return torch.stack(scores)  
    
    
    def step(self):  
        """执行一步优化"""  
        # 更新所有层的satisfaction scores  
        satisfaction_scores = self.update_satisfaction_scores()  
        
        # 使用scheduler分配新的预算  
        layer_budgets = self.scheduler(satisfaction_scores)  
        
        # 根据新的预算更新每一层  
        for layer_idx, budget in enumerate(layer_budgets):  
            self.compute_new_P_vectors(layer_idx, int(budget))  
            # 其他矩阵的更新...  



class EnhancedSVDReasoningAllocator:  
    def __init__(self, config):  
        self.config = config  
        
    def allocate_reasoning_budget(self, model_state, satisfaction_scores):  
        """主要分配逻辑"""  
        for layer_idx in range(self.config.num_layers):  
            # 获取当前层的SVD分解  
            P_k, Lambda_k, Q_k = self.get_layer_svd(  
                model_state,   
                layer_idx  
            )  
            
            # 计算需要增加的步骤数  
            delta_r = self.compute_delta_steps(  
                satisfaction_scores[layer_idx],  
                layer_idx  
            )  
            
            if delta_r > 0:  
                # 执行垂直扩展  
                P_new, Lambda_new, Q_new = self.vertical_expansion(  
                    P_k, Lambda_k, Q_k,  
                    delta_r,  
                    satisfaction_scores[layer_idx]  
                )  
                
                # 更新模型状态  
                self.update_layer_svd(  
                    model_state,  
                    layer_idx,  
                    P_new, Lambda_new, Q_new  
                )  
    
    def compute_delta_steps(self, satisfaction_score, layer_idx):  
        """计算需要增加的步骤数"""  
        base_steps = math.ceil(  
            self.config.step_increase_rate *   
            satisfaction_score *   
            math.log(layer_idx + 1)  
        )  
        return min(  
            base_steps,  
            self.config.max_steps - self.config.min_steps  
        )


    
    
    
    


class AdaptiveSingularValueEstimator:  
    def __init__(self, config: BudgetSchedulerConfig):  
        self.momentum = {}  # 每个奇异值的动量  
        self.velocity = {}  # 每个奇异值的速度  
        self.t = 0    # 时间步 
        self.config = config  
        
        '''
        :param: min_step: 每层的平均最小推理步数
        
        :param: input_length: 输入问题的长度，每层的最大推理步数为2 * input_length
        
        '''
        
        self.n_layers = config.n_layers

        # self.b_min = n_layers * min_steps  
        # self.b_max = n_layers * 2 * input_length  

        self.config = config  
        
        # 温度相关的状态变量  
        self.last_budget_grad = None  
        self.T0 = config.T0  
        self.last_budget = None 
        
    def compute_entropy(self, lambda_matrix):  
        '''
        
        计算奇异值矩阵Lambda的总熵 
        
        这个熵值用于所有奇异值的更新，反映了整个奇异值分布的不确定性
        
        lambda_matrix.shape = (n_step, n_step)
        
        '''
        # 计算每个奇异值的相对重要性熵  
        total = lambda_matrix.sum()  
        if total == 0:  
            return torch.zeros_like(lambda_matrix)  
        probs = lambda_matrix / total  
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))  
        return entropy  
         
    def exploration_term(self):  
        '''
        # 计算周期性探索项  
        # E(t) = A \cdot \sin^2(\omega t + \phi) \cdot \exp(-\lambda t)
        A: 振幅
        ω： 频率
        φ 是相位
        λ 是衰减率
        t 是时间步
        '''
        
        # 将所有变量转为张量，不然没法用torch.sin函数
        t = torch.tensor(self.t, dtype=torch.float32) / self.config.t_scale  
        omega = torch.tensor(self.config.omega, dtype=torch.float32)  
        phi = torch.tensor(self.config.phi, dtype=torch.float32)  
        A = torch.tensor(self.config.A, dtype=torch.float32)  
        lambda_ = torch.tensor(self.config.lambda_, dtype=torch.float32)  
        
        
        return (A * torch.sin(omega * t + phi)**2 * torch.exp(lambda_ * t))  
    
    def update_momentum(self, i, delta):  
        '''
        Args:
        :param: i: 奇异值的索引
        :param delta: delta_lambda_i^(t-1)
        '''
        if i not in self.momentum:  
            self.momentum[i] = 0  
            self.velocity[i] = 0  
            
        self.momentum[i] = (self.config.beta * self.momentum[i] +   
                           (1 - self.config.beta) * delta)  
        self.velocity[i] = (self.config.alpha * self.velocity[i] +   
                           (1 - self.config.alpha) * self.momentum[i]**2)  
    
    def compute_temperature(self):  
        """计算自适应温度  
        
        基于预算梯度的比值动态调整温度：  
        T^{(t)} = T_0 · exp(-η · ‖∇b^{(t)}‖₂/‖∇b^{(t-1)}‖₂)  
        """  
        if self.t < 2 or self.last_budget_grad is None:  
            return self.T0  
        
        # 计算当前预算的梯度  
        if hasattr(self, 'last_budget') and self.last_budget is not None:  
            current_grad = torch.abs(self.final_budget - self.last_budget)  
        else:  
            return self.T0  
        
        # 计算梯度比值  
        grad_ratio = current_grad / (self.last_budget_grad + 1e-8)  
        
        # 更新温度  
        temperature = self.T0 * torch.exp(-self.config.eta * grad_ratio)  
        
        # 保存当前梯度用于下次计算  
        self.last_budget_grad = current_grad  
        
        # 限制温度范围  
        temperature = torch.clamp(temperature,   
                                min=self.config.min_temperature,  
                                max=self.config.max_temperature)  
        
        return temperature  
    
    def compute_delta_lambda(self, lambda_matrix, i, layer_idx):  
        """
        计算单个奇异值的增量
        
        :param: i 奇异值的序号
        """  
        # 计算熵  
        entropy = self.compute_entropy(lambda_matrix)  
        
        # 计算探索项  
        explore = self.exploration_term()
        
        # 计算自适应因子  
        adaptive_factor = torch.sigmoid(entropy / self.config.H_max) + explore  
        
        # 计算差距项  
        lambda_max = lambda_matrix.max()  
        gap = lambda_max - lambda_matrix[i]  
        
        # 获取该层该索引的动量和速度  
        # # 使用元组作为key可以区分不同层的相同索引位置  
            # key1 = (0, 1)  # 第0层的第1个奇异值  
            # key2 = (1, 1)  # 第1层的第1个奇异值  
            
        # self.momentum和self.velocity的结构如下：  
            # self.momentum = {  
            #     (0, 0): 值,  # 第0层第0个奇异值的动量  
            #     (0, 1): 值,  # 第0层第1个奇异值的动量  
            #     (0, 2): 值,  # 第0层第2个奇异值的动量  
            #     (1, 0): 值,  # 第1层第0个奇异值的动量  
            #     (1, 1): 值,  # 第1层第1个奇异值的动量  
            #     # ... 以此类推  
            # }  
            
        key = (layer_idx, i)  
        if key not in self.momentum:  
            self.momentum[key] = 0  
            self.velocity[key] = 0  
            
        # 更新动量和速度  
        self.momentum[key] = (self.config.beta * self.momentum[key] +   
                            (1 - self.config.beta) * gap)  
        self.velocity[key] = (self.config.alpha * self.velocity[key] +   
                            (1 - self.config.alpha) * self.momentum[key]**2)  
        
        # 应用动量修正  
        # 类似Adam优化器的修正机制
        # 自适应调整更新步长
        momentum_correction = (self.momentum[key] /   
                             (torch.sqrt(self.velocity[key] + 1e-8)))  
        
        # 计算最终增量  
        delta = self.config.alpha * adaptive_factor * momentum_correction * gap  
        
        # alpha：基础学习率
        # adaptive_factor：自适应调整因子
        # momentum_correction：动量修正
        # gap：与最大奇异值的差距
        
        # 好处
        # 重要的奇异值得到加强
        # 更新过程平滑且自适应
        # 考虑历史信息（通过动量）
        # 具有探索能力
        return delta  
    
    
    
    def compute_total_budget(self, lambda_matrices):  
        """计算整个网络的总奇异值预算b^(t)"""  
        # 计算所有层的总熵  
        total_entropy = 0  
        total_singular_values = 0  
        
        for lambda_matrix in lambda_matrices.values():  
            total_entropy += self.compute_entropy(lambda_matrix)  
            total_singular_values += len(lambda_matrix)  
            
        # 计算预算比例  
        budget_ratio = torch.sigmoid(total_entropy / (total_singular_values * self.config.H_max))  
        
        # 添加探索项  
        # explore = (self.config.A *   
        #           torch.sin(self.config.omega * self.t + self.config.phi)**2 *   
        #           torch.exp(-self.config.lambda_ * self.t))  
        
        explore = self.exploration_term()
        
        # 计算总预算  
        total_budget = int(self.config.min_rank +   
                          (self.config.max_rank - self.config.min_rank) *   
                          (budget_ratio + explore).item())  
        
        return total_budget  
    
    def allocate_layer_budgets(self, lambda_matrices, total_budget):  
        """将总预算分配到各层  
        
        Args:  
            lambda_matrices: 各层的奇异值矩阵  
            total_budget: 总预算  
        Returns:  
            Dict[int, int]: 每层分配的预算  
        """  
        layer_weights = {}  
        total_weight = 0  
        
        # 计算每层的权重（基于熵和奇异值分布）, 我们根据权重的大小来分配预算  
        for layer_idx, lambda_matrix in lambda_matrices.items():  
            entropy = self.compute_entropy(lambda_matrix)  
            max_singular_value = lambda_matrix.max()  
            
            # 权重 = 每层的熵 * 每层最大奇异值  
            weight = entropy * max_singular_value  
            layer_weights[layer_idx] = weight  
            total_weight += weight  
        
        # 按权重分配预算  
        layer_budgets = {}  
        remaining_budget = total_budget  
        
        for layer_idx in lambda_matrices.keys():  
            if layer_idx == len(lambda_matrices) - 1:  
                # 最后一层获得剩余预算  
                layer_budgets[layer_idx] = remaining_budget  
            else:  
                # 按权重比例分配  
                budget = int((layer_weights[layer_idx] / total_weight) * total_budget)  
                layer_budgets[layer_idx] = budget  
                remaining_budget -= budget  
        
        return layer_budgets  
    
    
    def update_layer_singular_values(self, new_lambda, layer_idx, layer_budget, satisfaction_scores):  
        """更新单层的奇异值  
        
         按照预算分配更新奇异值:  
            λᵢ^(l,t+1) = λᵢ^(l,t) + Δλᵢ^(l,t)  如果 λᵢ^(l,t) 是前 b_l^(t) 大的奇异值  
            λᵢ^(l,t+1) = λᵢ^(l,t)               否则  
    
        Args:  
            new_P: 经过FixedSVD扩展后的P矩阵  
            new_Q: 经过FixedSVD扩展后的Q矩阵  
            new_lambda: 经过FixedSVD扩展后的奇异值矩阵  
            
            layer_idx: 层索引  
            layer_budget: 该层分配到的预算  
            fixed_rank_svd: FixedRankSVD对象，用于处理SVD分解相关的操作  
            
            satisfaction_scores: layer 的满意度分数
        
        Returns:  
            torch.Tensor: 更新后的奇异值矩阵  
        """  
        # 获取当前维度  
        current_dim = new_lambda.size(0)  
        
        # 1. 获取当前奇异值向量（对角线元素）  
        current_singular_values = torch.diag(new_lambda)  
        
        # 计算需要更新的奇异值数量（b_l^(t)/2）  
        update_count = max(1, int(layer_budget // 2))  # 确保至少更新1个  
        
        # 2. 找出前layer_budget大的奇异值索引  
        _, top_indices = torch.topk(current_singular_values,   
                                k=min(update_count, current_dim))  
        
        
        # 1. 首先通过FixedRankSVD扩展P、Lambda、Q矩阵  
        # new_P = fixed_rank_svd.compute_new_P_vectors(layer_idx, layer_budget)
        # new_Q = fixed_rank_svd.compute_new_Q_vectors(layer_idx, layer_budget)
        # new_lambda = fixed_rank_svd.compute_new_Lambda_vectors(layer_idx, layer_budget)  
        
        # n_l = lambda_matrix.size(0)  # 该层的奇异值数量  
        # new_lambda = lambda_matrix.clone()  
        
        # 获取最大的layer_budget个奇异值的索引 
        
        # 这里有问题， 这里 min(layer_budget, n_l) 确保了每层被分配的预算不会超过该层原有的奇异值数量
        # 而我们之前的设计是拓展奇异值矩阵，所以这里新的的n_l直接等于layer_budget
        # 
        # _, top_indices = torch.topk(lambda_matrix, k=min(layer_budget, n_l))  
        
        # 使用扩展后的新维度更新奇异值   
        # for i in range(layer_budget):  
        #     delta = self.compute_delta_lambda(new_lambda, i, layer_idx)  
        #     new_lambda[i] = new_lambda[i] + delta  
        
        
        # 更新奇异值  
        # 创建更新掩码（mask）  
        update_mask = torch.zeros_like(current_singular_values, dtype=torch.bool)  
        update_mask[top_indices] = True  
        
        # 对每个需要更新的奇异值计算增量并更新  
        for i in range(new_lambda.size(0)):  
            if update_mask[i]:  
                # 只对满意度分数较高的奇异值计算和应用增量  
                delta = self.compute_delta_lambda(new_lambda, i, layer_idx)  
                new_lambda[i, i] = new_lambda[i, i] + delta  
        
        return new_lambda 
    
    def update_network_singular_values(self, lambda_matrices):  
        """更新整个网络的奇异值  
        
        Args:  
            lambda_matrices: Dict[int, torch.Tensor] 每层的奇异值矩阵  
                key: 层索引  
                value: 该层的奇异值向量 shape=(n_l,)，其中n_l是该层的奇异值数量  
        Returns:  
            Dict[int, torch.Tensor]: 更新后的各层奇异值矩阵  
        """  
        
        
        
        # 1. 计算总预算b^(t)  
        total_budget = self.compute_total_budget(lambda_matrices)  
        
        # 2. 计算各层的重要性权重并分配预算  
        layer_budgets = self.allocate_layer_budgets(lambda_matrices, total_budget)  
        
        # 3. 更新各层的奇异值  
        new_lambda_matrices = {}  
        for layer_idx, lambda_matrix in lambda_matrices.items():  
            
            satisfaction_scores = self.compute_satisfaction_scores(lambda_matrix)
            new_lambda = self.update_layer_singular_values(  
                lambda_matrix,  
                layer_idx,  
                layer_budgets[layer_idx]  
            )  
            new_lambda_matrices[layer_idx] = new_lambda  
            
        self.t += 1  
        return new_lambda_matrices  
        
    def __call__(self, lambda_matrix):  
        # 计算熵 
        # 计算奇异值矩阵的总熵 H_i^t
        entropy = self.compute_entropy(lambda_matrix)  

        # 计算探索项  
        explore = self.exploration_term()  
        
        # 计算预算  
        
        # 计算预算比例， 实际上约等于模型中的层数
        budget_ratio = torch.sigmoid(entropy / self.config.H_max)  
        
        # b^(t), 但是还没有乘动量平滑
        base_budget = int(self.config.min_rank +   
                         (self.config.max_rank - self.config.min_rank) *   
                         (budget_ratio + explore).item())  
        
        # 更新奇异值  
        new_lambda = lambda_matrix.clone()  
        lambda_max = lambda_matrix.max()  
        
        for i in range(len(lambda_matrix)):   # for i in range(n_step)
            if i < base_budget:  
                # 计算增量  delta_lambda_i^(t)
                importance = torch.sigmoid(entropy / self.config.H_max)  
                delta = self.config.alpha * (importance + explore) * (lambda_max - lambda_matrix[i])  
                
                # 更新动量 ， 更新速度
                self.update_momentum(i, delta)  
                
                # 应用动量修正  
                final_delta = delta * (self.momentum[i] /   
                                     (torch.sqrt(self.velocity[i]) + 1e-8))  
                
                # 更新奇异值  
                new_lambda[i] = lambda_matrix[i] + final_delta  
        
        self.t += 1  
        return new_lambda  





# 测试代码  
def test_singular_value_update():  
    # 1. 创建配置  
    config = BudgetSchedulerConfig()  
    
    # 2. 创建更新器  
    updater = AdaptiveSingularValueEstimator(config)  
    
    # 3. 创建测试用的奇异值矩阵  
    lambda_matrices = {  
        0: torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),  # 第一层  
        1: torch.tensor([4.0, 3.0, 2.0, 1.0]),       # 第二层  
        2: torch.tensor([3.0, 2.0, 1.0])             # 第三层  
    }  
    
    # 4. 打印初始状态  
    print("Initial singular values:")  
    for layer_idx, lambda_matrix in lambda_matrices.items():  
        print(f"Layer {layer_idx}: {lambda_matrix}")  
    
    # 5. 运行多个时间步  
    n_steps = 5  
    for step in range(n_steps):  
        # 更新奇异值  
        new_lambda_matrices = updater.update_network_singular_values(lambda_matrices)  
        
        # 打印结果  
        print(f"\nStep {step + 1}:")  
        print("Total budget:", updater.compute_total_budget(lambda_matrices))  
        
        # 打印每层的预算分配  
        layer_budgets = updater.allocate_layer_budgets(  
            lambda_matrices,   
            updater.compute_total_budget(lambda_matrices)  
        )  
        print("Layer budgets:", layer_budgets)  
        
        # 打印更新后的奇异值  
        for layer_idx, new_lambda in new_lambda_matrices.items():  
            print(f"Layer {layer_idx}: {new_lambda}")  
        
        # 更新lambda_matrices用于下一步  
        lambda_matrices = new_lambda_matrices




















if __name__ == '__main__':
    # config = {  
    #     'hidden_size': 768,  
    #     'prefix_length': 10,  
    #     'suffix_length': 10,  
    #     'num_layers': 2,  
    #     'dropout': 0.1  
    # } 
    
    # # 初始化模型  
    # prompt_encoder =BaasPromptEncoder(**config)  
    
    # # 前向传播  
    # batch_size = 4  
    # prefix_embeddings, suffix_embeddings = prompt_encoder(batch_size)  
    
    # # 可视化注意力权重（如果需要）  
    # prefix_to_suffix_attention = prompt_encoder.prefix_to_suffix_attention.attention_weights  
    # suffix_to_prefix_attention = prompt_encoder.suffix_to_prefix_attention.attention_weights  
    
    
    
    
    # print(prefix_to_suffix_attention)
    # print(suffix_to_prefix_attention)
    
    
    
    
    # 使用示例  
    # config = Config(  
    #     min_rank=10,  
    #     max_rank=100,  
    #     alpha=0.01,  
    #     beta=0.9,  
    #     H_max=1.0,  
    #     A=0.1,  
    #     omega=0.1,  
    #     phi=0,  
    #     lambda_=0.001,  
    #     t_scale=1000  
    # )  
    
    # config = BudgetSchedulerConfig()
    
    

    # estimator = AdaptiveSingularValueEstimator(config)  

    # # 在训练循环中使用  
    # lambda_matrix = torch.randn(100)  # 初始奇异值  
    # new_lambda = estimator(lambda_matrix)
    
    
    # print(new_lambda)
    
    
    
    
    
    
    
    test_singular_value_update()