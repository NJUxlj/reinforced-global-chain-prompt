import torch
import torch.nn as nn
import torch.nn.functional as F  

from transformers import AutoModel, AutoTokenizer
from config import Config   

from typing import List, Tuple
import math

'''

this file contains various model components, such as:

prompt encoder, cross-attention, gated-cross-attention, sparse-attention ...


'''


class InputEncoder(nn.Module):
    '''
    function:
        1. compress the length of the  real input sequence into [max_seq_length - prefix_length - suffix_length - hard_prompt_length] 
        2. 使用SparseAttention来将MCQ的input压缩到[max_seq_length - prefix_length - suffix_length - hard_prompt_length]
    '''
    def __init__(self, target_length, hidden_size):
        pass




class PromptEncoder(nn.Module):
    
    '''
    function:
        1.使用 SparseAttention来编码MCQ多项选择数据集的input
        
    
    '''
    def __init__(self, template, hidden_size, tokenizer, device, args): 
        pass





class GatedMultiHeadCrossAttention(nn.Module):  
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):  
        super().__init__()  
        self.hidden_size = hidden_size  
        self.num_heads = num_heads  
        self.head_dim = hidden_size // num_heads  

        
        # Multi-head projections  
        self.q_proj = nn.Linear(hidden_size, hidden_size)  # W^Q 
        self.k_proj = nn.Linear(hidden_size, hidden_size)  # W^K
        self.v_proj = nn.Linear(hidden_size, hidden_size)  # W^V
        self.o_proj = nn.Linear(hidden_size, hidden_size)  
        
        # Gating mechanism  
        self.gate = nn.Sequential(  
            nn.Linear(hidden_size * 2, hidden_size),  
            nn.Sigmoid()  
        )  
       
       # Layer normalization  
        self.norm1 = nn.LayerNorm(hidden_size)  
        self.norm2 = nn.LayerNorm(hidden_size)  
        
        # Dropout  
        self.dropout = nn.Dropout(dropout) 

    
    def forward(self, reasoning_matrices):
        # matrices: [K, K_suffix, H]  
        batch_size = reasoning_matrices.size(0)  


        # Learnable query  
        query = self.q_proj(reasoning_matrices)  # [K, K_suffix, H]  
        key = self.k_proj(reasoning_matrices)    # [K, K_suffix, H]  
        value = self.v_proj(reasoning_matrices)  # [K, K_suffix, H]  


        # Reshape for multi-head attention 


        # Scaled dot-product attention
        
        
        
        # Apply attention to values

        
        # Output projection
        
        
        
        # Gating mechanism 
        
        
        
        # Residual connection and layer norm 







class SparseAttention(nn.Module):
    '''
    Adaptive Sparse Attention with Local-Global Information
    '''
    def __init__(self, hidden_size, num_heads=8, local_window=3, sparsity_threshold=0.1): 
        pass
    
    
    
    









class BidirectionalAttentionalPromptEncoder(nn.Module):  
    def __init__(  
        self,   
        hidden_size: int,  
        prefix_length: int,  
        suffix_length: int,  
        num_layers: int = 2,  
        dropout: float = 0.1  
    ):  
        super().__init__()  
        self.hidden_size = hidden_size  
        self.prefix_length = prefix_length  
        self.suffix_length = suffix_length  
        
        # 1. 可学习的原始嵌入  
        self.prefix_embeddings = nn.Parameter(  
            torch.randn(prefix_length, hidden_size)  
        )  
        self.suffix_embeddings = nn.Parameter(  
            torch.randn(suffix_length, hidden_size)  
        )  
        
        # 2. 双向LSTM编码器  
        self.forward_lstm = nn.LSTM(  
            hidden_size,  
            hidden_size // 2,  
            num_layers=num_layers,  
            bidirectional=True,  
            batch_first=True  
        )  
        self.backward_lstm = nn.LSTM(  
            hidden_size,  
            hidden_size // 2,  
            num_layers=num_layers,  
            bidirectional=True,  
            batch_first=True  
        )  
        
        # 3. 交互注意力层  
        self.prefix_to_suffix_attention = MultiHeadCrossAttention(  
            hidden_size, num_heads=8  
        )  
        self.suffix_to_prefix_attention = MultiHeadCrossAttention(  
            hidden_size, num_heads=8  
        )  
        
        # 4. 自适应门控融合  
        self.prefix_gate = AdaptiveGate(hidden_size)  
        self.suffix_gate = AdaptiveGate(hidden_size)  
        
        # 5. 位置编码  
        self.prefix_pos_embedding = SinusoidalPositionalEmbedding(hidden_size)  
        self.suffix_pos_embedding = SinusoidalPositionalEmbedding(hidden_size)  
        
        # 6. 输出转换层  
        self.prefix_output_layer = OutputTransformation(hidden_size)  
        self.suffix_output_layer = OutputTransformation(hidden_size)  
        
        self.dropout = nn.Dropout(dropout)  
        self.layer_norm = nn.LayerNorm(hidden_size)  

    def forward(self, batch_size: int = 1):  
        # 1. 扩展batch维度  
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        suffix_embeds = self.suffix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        
        # 2. 添加位置编码  
        prefix_embeds = self.prefix_pos_embedding(prefix_embeds)  
        suffix_embeds = self.suffix_pos_embedding(suffix_embeds)  
        
        # 3. 双向LSTM编码  
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
        
        # 4. 交互注意力  
        prefix_attended = self.prefix_to_suffix_attention(  
            prefix_bidirectional, suffix_bidirectional, suffix_bidirectional  
        )  
        suffix_attended = self.suffix_to_prefix_attention(  
            suffix_bidirectional, prefix_bidirectional, prefix_bidirectional  
        )  
        
        # 5. 残差连接和归一化  
        prefix_output = self.layer_norm(prefix_bidirectional + prefix_attended)  
        suffix_output = self.layer_norm(suffix_bidirectional + suffix_attended)  
        
        # 6. 最终转换  
        prefix_output = self.prefix_output_layer(prefix_output)  
        suffix_output = self.suffix_output_layer(suffix_output)  
        
        return prefix_output, suffix_output  
    



class MultiHeadCrossAttention(nn.Module):  
    def __init__(self, hidden_size, num_heads=8):  
        super().__init__()  
        self.num_heads = num_heads  
        self.head_dim = hidden_size // num_heads  
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)  
        self.k_proj = nn.Linear(hidden_size, hidden_size)  
        self.v_proj = nn.Linear(hidden_size, hidden_size)  
        self.o_proj = nn.Linear(hidden_size, hidden_size)  
        
        self.attention_weights = None  # 存储注意力权重用于可视化  
        
    def forward(self, query, key, value):  
        B, N, C = query.shape  
        
        # 多头投影  
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim)  
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim)  
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim)  
        
        # 注意力计算  
        attn = torch.einsum('bnhd,bmhd->bnmh', q, k) / math.sqrt(self.head_dim)  
        attn = F.softmax(attn, dim=2)  
        self.attention_weights = attn  # 保存注意力权重  
        
        # 输出  
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v)  
        out = out.reshape(B, N, -1)  
        out = self.o_proj(out)  
        
        return out  
    
    
    



class AdaptiveGate(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.gate_network = nn.Sequential(  
            nn.Linear(hidden_size * 2, hidden_size),  
            nn.LayerNorm(hidden_size),  
            nn.GELU(),  
            nn.Linear(hidden_size, hidden_size * 2),  
            nn.Sigmoid()  
        )  
        
    def forward(self, forward_features, backward_features):  
        # 计算门控权重  
        combined = torch.cat([forward_features, backward_features], dim=-1)  
        gates = self.gate_network(combined)  
        forward_gate, backward_gate = gates.chunk(2, dim=-1)  
        
        # 门控融合  
        output = forward_gate * forward_features + backward_gate * backward_features  
        return output  
    




class SinusoidalPositionalEmbedding(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.hidden_size = hidden_size  
        
    def forward(self, x):  
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(-1)  
        dim_pos = torch.arange(0, self.hidden_size, 2, device=x.device)  
        div_term = torch.exp(-math.log(10000.0) * dim_pos / self.hidden_size)  
        
        pos_embedding = torch.zeros(x.size(1), self.hidden_size, device=x.device)  
        pos_embedding[:, 0::2] = torch.sin(positions * div_term)  
        pos_embedding[:, 1::2] = torch.cos(positions * div_term)  
        
        return x + pos_embedding.unsqueeze(0)





class OutputTransformation(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.transform = nn.Sequential(  
            nn.Linear(hidden_size, hidden_size * 2),  
            nn.LayerNorm(hidden_size * 2),  
            nn.GELU(),  
            nn.Linear(hidden_size * 2, hidden_size),  
            nn.LayerNorm(hidden_size)  
        )  
        
    def forward(self, x):  
        return self.transform(x) 






if __name__ == '__main__':
    config = {  
        'hidden_size': 768,  
        'prefix_length': 10,  
        'suffix_length': 10,  
        'num_layers': 2,  
        'dropout': 0.1  
    } 
    
    # 初始化模型  
    prompt_encoder = BidirectionalAttentionalPromptEncoder(**config)  
    
    # 前向传播  
    batch_size = 4  
    prefix_embeddings, suffix_embeddings = prompt_encoder(batch_size)  
    
    # 可视化注意力权重（如果需要）  
    prefix_to_suffix_attention = prompt_encoder.prefix_to_suffix_attention.attention_weights  
    suffix_to_prefix_attention = prompt_encoder.suffix_to_prefix_attention.attention_weights  
    
    
    
    
    print(prefix_to_suffix_attention)
    print(suffix_to_prefix_attention)