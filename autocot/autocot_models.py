import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math
from typing import Optional, Tuple, List, Dict, Any, Type
import sys
import os

# 获取当前文件所在目录的父目录  
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# 将父目录添加到sys.path  
sys.path.insert(0, parent_directory) 
from config import Config

class RoutedCrossAttention(nn.Module):  
    def __init__(  
        self,  
        hidden_size: int,  
        num_attention_heads: int = 8,  
        dropout: float = 0.1,  
        attention_dropout: float = 0.1  
    ):  
        """  
        
        Args:  
            hidden_size: 隐藏层维度  
            num_attention_heads: 注意力头数量  
            dropout: dropout率  
            attention_dropout: 注意力dropout率  
        """  
        super().__init__()  
        self.hidden_size = hidden_size  
        self.num_attention_heads = num_attention_heads  
        self.attention_head_size = hidden_size // num_attention_heads  
        self.all_head_size = self.num_attention_heads * self.attention_head_size  
        
        # 定义投影矩阵  
        self.query = nn.Linear(hidden_size, self.all_head_size)  
        self.key = nn.Linear(hidden_size, self.all_head_size)  
        self.value = nn.Linear(hidden_size, self.all_head_size)  
        
        # 输出投影  
        self.output = nn.Linear(hidden_size, hidden_size)  
        
        # Dropout  
        self.dropout = nn.Dropout(dropout)  
        self.attention_dropout = nn.Dropout(attention_dropout)  
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:  
        """将张量转换为多头注意力格式""" 
        # new_x_shape = (K, min_ra_len, hidden_size) 
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  
        x = x.view(*new_x_shape)  
        return x.permute(0, 2, 1, 3)  
        
    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None  
    ) -> torch.Tensor:  
        """  
        前向传播  
        
        Args:  
            hidden_states: 输入张量，shape=(K, min_ra_len, hidden_size)  
            attention_mask: 注意力掩码，可选  
            
        Returns:  
            torch.Tensor: 融合后的上下文表示，shape=(min_ra_len, hidden_size)  
        """  
        # 计算Q、K、V  
        query_layer = self.transpose_for_scores(self.query(hidden_states))  
        key_layer = self.transpose_for_scores(self.key(hidden_states))  
        value_layer = self.transpose_for_scores(self.value(hidden_states))  
        
        # 计算注意力分数  
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  
        attention_scores = attention_scores / torch.sqrt(  
            torch.tensor(self.attention_head_size, dtype=torch.float)  
        )  
        
        # 应用注意力掩码（如果有）  
        if attention_mask is not None:  
            attention_scores = attention_scores + attention_mask  
            
        # 注意力权重  
        attention_probs = F.softmax(attention_scores, dim=-1)  
        attention_probs = self.attention_dropout(attention_probs)  
        
        # 计算上下文向量  
        context_layer = torch.matmul(attention_probs, value_layer)  
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  
        context_layer = context_layer.view(*new_context_layer_shape)  
        
        # 输出投影  
        output = self.output(context_layer)  
        output = self.dropout(output)  
        
        # 融合多个embeddings为一个context  
        # 使用注意力权重的平均值作为融合权重  
        fusion_weights = attention_probs.mean(dim=(1, 2))  # shape=(K, min_ra_len)  
        fused_context = torch.sum(  
            output * fusion_weights.unsqueeze(-1),  
            dim=0  
        )  
        
        return fused_context  


class MultiHeadAttention(nn.Module):    
    def __init__(  
        self,  
        hidden_size: int,  
        num_heads: int = 8,  
        dropout: float = 0.1,  
        bias: bool = True,
        device = Config.device
    ):  
        super().__init__()  
        self.hidden_size = hidden_size  
        self.num_heads = num_heads  
        self.device = device
        self.dropout = dropout  
        self.head_size = hidden_size // num_heads  
        self.scaling = self.head_size ** -0.5    # 1 / sqrt(d_k)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias).to(device)  
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias).to(device)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias).to(device)  
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias).to(device)  
        
        self.dropout_p = nn.Dropout(dropout).to(device)  
        
        # 确保hidden_size能被num_heads整除  
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}" + \
                f", and currently, hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
        
    def forward(  
        self,  
        query: torch.Tensor,      # (B, L, H)  
        key: torch.Tensor,        # (B, S, H)  
        value: torch.Tensor,      # (B, S, H)  
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)  防止模型去关注那些被填充的位置
        attn_mask: Optional[torch.Tensor] = None         # (L, S) or (B, L, S)  
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """  
        Args:  
            query: (batch_size, query_length, hidden_size)  
            key: (batch_size, key_length, hidden_size)  
            value: (batch_size, key_length, hidden_size)  
            key_padding_mask: (batch_size, key_length)  
            attn_mask: (query_length, key_length) or (batch_size, query_length, key_length)  
            
        Returns:  
            output: (batch_size, query_length, hidden_size)  
            attention_weights: (batch_size, num_heads, query_length, key_length)  
        """  
        batch_size, query_length, hidden_size = query.size()  
        key_length = key.size(1)  
        
        q_batch_size, k_batch_size, v_batch_size = query.size(0), key.size(0), value.size(0)
        
        # 1. 线性投影并重塑为多头形式  
        
        # q_proj.shape = (hidden_size, hidden_size)
        print("q_batch_size = ", q_batch_size)
        print("k_batch_size = ", k_batch_size)
        print("v_batch_size = ", v_batch_size)
        print("key_length = ", key_length)
        print("real hidden_size = ", hidden_size)
        print("hidden_size = ", self.q_proj.out_features)
        print("self.num_heads x self.head_size = ", self.num_heads * self.head_size)
        
        
        q = self.q_proj(query).view(q_batch_size, query_length, self.num_heads, self.head_size).to(self.device)
        k = self.k_proj(key).view(k_batch_size, key_length, self.num_heads, self.head_size).to(self.device)
        v = self.v_proj(value).view(v_batch_size, key_length, self.num_heads, self.head_size).to(self.device)  
        
        # 调整维度顺序为 (batch_size, num_heads, length, head_size), 因此只有最后两维可以做矩阵乘法
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  
        
        # 2. 计算注意力分数
        # shape = (max(q_batch_size, k_batch_size), num_heads, query_length, key_length)
        attn_weights:torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  
        
        
        # 应用掩码(如果有)  
        if attn_mask is not None:  
            if attn_mask.dim() == 2:  
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  
            elif attn_mask.dim() == 3:  
                attn_mask = attn_mask.unsqueeze(1)  
            attn_weights += attn_mask  
        
        if key_padding_mask is not None:  
            attn_weights = attn_weights.masked_fill(  
                key_padding_mask.unsqueeze(1).unsqueeze(2),   # 扩展为 [batch_size, 1, 1, key_length], 再广播到attn_weights的尺寸
                float('-inf')  
            )  
        
        # 4. Softmax归一化  
        attn_weights = F.softmax(attn_weights, dim=-1)  
        attn_weights = self.dropout_p(attn_weights)  
        
        # 5. 注意力加权求和  
        output = torch.matmul(attn_weights, v)  
        
        # 6. 重塑回原始维度  
        output = output.transpose(1, 2).contiguous().view(  
            max(q_batch_size, k_batch_size, v_batch_size), query_length, self.hidden_size  
        )  
        
        # 7. 最终的线性投影  
        output = self.out_proj(output)  
        
        return output, attn_weights  
  
  
class FeedForward(nn.Module):  
    """前馈神经网络层"""  
    def __init__(  
    self,  
    hidden_size: int,  
    ff_size: int = None,  
    dropout: float = 0.1,  
    activation: str = 'relu',
    device = Config.device 
    ):  
        super().__init__()  
        ff_size = ff_size or hidden_size * 4  
        
        self.ff_layer = nn.Sequential(  
            nn.Linear(hidden_size, ff_size),  
            nn.ReLU() if activation == 'relu' else nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(ff_size, hidden_size)  
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.ff_layer(x)  



class OriginalHAN(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        # 词级注意力  
        self.word_attention = MultiHeadAttention(hidden_size)  
        # 句子级注意力  
        self.sentence_attention = MultiHeadAttention(hidden_size)  
        
    def forward(self, document):  
        # document: (num_sentences, num_words, hidden_size)  
        
        # 词级注意力  
        sentence_vectors = []  
        
        # sentence.shape = (num_words, hidden_size)
        for sentence in document:  
            # 对每个句子中的词进行self-attention计算, 实际上就是对句子本身做自注意力
            # word_weights.shape = (num_words, hidden_size)
            word_weights = self.word_attention(sentence)  
            
            # sentence_vec.shape = (num_words,)
            
            sentence_vec = (word_weights * sentence).sum(dim=1)  
            sentence_vectors.append(sentence_vec)  
            
        # 句子级注意力  
        document_vector = self.sentence_attention(  
            torch.stack(sentence_vectors)   # shape = (num_sentences, num_words)
        )  # shape = (num_sentences, num_words)
        return document_vector
    
    
    
class HierarchicalAttentionFusion(nn.Module):  
    """基于层次化注意力的思维链融合模型"""  
    def __init__(  
        self,  
        hidden_size: int,  
        num_heads: int = 8,  
        dropout: float = 0.1,  
        activation: str = 'relu',  
        num_layers: int = 2,  
        use_ffn: bool = True,
        device = Config.device
    ):  
        super().__init__()  
        self.hidden_size = hidden_size
        self.device = device  
        self.num_heads = num_heads  
        self.num_layers = num_layers  
        self.use_ffn = use_ffn  
        
        # 序列级别的注意力层  类似于Word Attention
        # 通过12层的自注意力机制来建模 某一条推理链的表征(chain context)
        self.sequence_attentions = nn.ModuleList([  
            MultiHeadAttention(  
                hidden_size=hidden_size,  
                num_heads=num_heads,  
                dropout=dropout,
            ) for _ in range(num_layers)  
        ]).to(device)  
        
        # 链级别的注意力层   类比于Sentence Attention  
        self.chain_attention = MultiHeadAttention(  
            hidden_size=hidden_size,  
            num_heads=num_heads,  
            dropout=dropout  
        ).to(device)  
        
        # Layer Normalization  
        self.sequence_norms = nn.ModuleList([  
            nn.LayerNorm(hidden_size) for _ in range(num_layers)  
        ]).to(device)  
        self.chain_norm = nn.LayerNorm(hidden_size).to(device)   
        
        # 前馈网络(可选)  
        if use_ffn:  
            self.sequence_ffns = nn.ModuleList([  
                FeedForward(  
                    hidden_size=hidden_size,  
                    dropout=dropout,  
                    activation=activation  
                ) for _ in range(num_layers)  
            ]).to(device)   
            self.chain_ffn = FeedForward(  
                hidden_size=hidden_size,  
                dropout=dropout,  
                activation=activation  
            ).to(device)   
            self.ffn_norms = nn.ModuleList([  
                nn.LayerNorm(hidden_size) for _ in range(num_layers + 1)  
            ]).to(device)   
        
        self.dropout = nn.Dropout(dropout).to(device)   
        
    def forward(  
        self,  
        embeddings: torch.Tensor,  
        padding_mask: Optional[torch.Tensor] = None  
    ) -> Tuple[torch.Tensor, dict]:  
        """  
        Args:  
            embeddings: shape (K, L, H) 其中:  
                K: 推理链数量  (类比于文档中的句子数)  
                L: 序列长度(min_ra_len)  (类比于句子中的词数)  
                H: hidden_size  
            padding_mask: shape (K, L), True表示需要mask的位置  
            
        Returns:  
            context: shape (L, H)  
            attention_info: 包含注意力权重的字典  
        """  
        embeddings = embeddings.to(self.device) 
        batch_size, seq_len, hidden_size = embeddings.size()  
        attention_info = {}  
        
        # 序列级别的注意力   处理每条推理链内的token(reasoning steps)关系
        '''
        序列级注意力(第一层)

            输入：每条推理链的token序列 (L个token)
            目的：捕获单条推理链内部的token之间的关系
            类比：类似于原论文中的Word Attention，但处理的是推理步骤而不是词
        '''
        sequence_output = embeddings   # shape (K, L, H)
 
        # 获取一个12层Attention建模后的 K个推理链 (embeddings) 表示
        for i in range(self.num_layers):  
            # Self-attention  
            # attn_output.shape = (K, L, H)
            # attn_weights.shape = (K, L, L)
            
            # self-attention:  q,k,v 全部来自同一个输入
            # why self-attention? 
            attn_output, attn_weights = self.sequence_attentions[i](
                query=sequence_output,
                key=sequence_output,
                value=sequence_output,
                key_padding_mask=padding_mask
            )  
            attention_info[f'sequence_attention_layer_{i}'] = attn_weights  
            
            # 残差连接和层归一化  
            sequence_output = self.sequence_norms[i](
                sequence_output+self.dropout(attn_output)  
            )  
            
            # 前馈网络(如果启用)  
            if self.use_ffn:  
                ff_output = self.sequence_ffns[i](sequence_output)  
                # 再次残差链接 + 归一化
                sequence_output = self.ffn_norms[i](
                    sequence_output+self.dropout(ff_output)  
                )  
        
        
        '''
        # 链级别的注意力 
            输入：所有推理链的表示 (K条链)
            目的：融合不同推理链的信息，找到最重要的推理路径
            类比：类似于原论文中的Sentence Attention，但处理的是推理链而不是句子
        '''
        # 使用第一条链作为query  (主链)
        chain_query = sequence_output[0:1]  # (1, L, H)  
        # chain_query = sequence_outputs[0].unsqueeze(0)
        
        # cross-attention: # query来自x，key和value来自context 
        # why cross attention: 我们想让主链去关注到其他所有链的内容 (建模链间关系)
        chain_output, chain_attn_weights = self.chain_attention(  
            query=chain_query,   #  (1, L, H)  
            key=sequence_output,   # (K, L, H)
            value=sequence_output,  # (K, L, H)
            key_padding_mask=padding_mask  
        )  # shape = (1, L, H)   
        '''
        批量矩阵乘法会在保持批次维度的同时，  
        对最后两个维度进行普通矩阵乘法  
        # 例子  
        A = torch.randn(8, 1, 10, 20)  # (batch, m1, m2, k)  
        B = torch.randn(8, 5, 20, 30)  # (batch, n1, k, n2)  
        
        # 结果: (8, 1, 10, 30)  
        # - 8(batch)保持不变  
        # - 1和5不参与计算，保留较小的1  
        # - 10和30来自外部维度  
        C = torch.matmul(A, B)
        
        总之， 输出的sequence长度由query决定

        '''
        
        print("chain_output.shape = ",chain_output.shape)
        attention_info['chain_attention'] = chain_attn_weights   # (1, L, L)
        
        # 残差连接和层归一化  
        chain_output = self.chain_norm(  
            chain_query + self.dropout(chain_output)  
        )  # shape = (1, L, H)
        
        # 链级别的前馈网络(如果启用)  
        if self.use_ffn:  
            chain_ff_output = self.chain_ffn(chain_output)  
            chain_output = self.ffn_norms[-1](
                chain_output+self.dropout(chain_ff_output)  
            )  
            
        # (LxH)
        return chain_output.squeeze(0), attention_info  


if __name__ == '__main__':  
    pass