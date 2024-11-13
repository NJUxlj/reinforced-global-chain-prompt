import torch
import torch.nn as nn
import torch.nn.functional as F  

from transformers import AutoModel, AutoTokenizer
from config import Config, BudgetSchedulerConfig   

from typing import List, Tuple, Dict, Optional, Any, Union
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




def compute_satisfaction_score_v1(k, i, P, Q, λ, l_q, L, d1, d2):  
    """  
    方案1: Layer-Weighted Gradient Impact Score  
    
    参数:  
    k: 当前层索引  
    i: 当前奇异值索引  
    P, Q: 奇异向量矩阵  
    λ: 奇异值  
    l_q: 问题长度  
    L: 总层数  
    d1, d2: P和Q的维度  
    """  
    # 层深度权重  
    layer_weight = (k + 1) / L  # 越深层权重越大  
    
    # 问题长度影响因子  
    length_factor = math.log(1 + l_q/30)  # 30是基准长度  
    
    # 基础梯度敏感度计算  
    s_lambda = abs(λ[i] * λ[i].grad) if λ[i].grad is not None else 0  
    s_P = torch.sum(abs(P[:, i] * P[:, i].grad)) / d1 if P[:, i].grad is not None else 0  
    s_Q = torch.sum(abs(Q[i, :] * Q[i, :].grad)) / d2 if Q[i, :].grad is not None else 0  
    
    # 组合梯度敏感度  
    gradient_impact = (s_lambda + s_P + s_Q) / 3  
    
    # 最终satisfaction score  
    score = layer_weight * length_factor * gradient_impact  
    
    return score 




def compute_satisfaction_score_v2(k, i, P, Q, Lambda, l_q, L, d1, d2, alpha=0.5):  
    """  
    额外参数：  
    α: 控制层深度影响的超参数  
    """  
    # 指数层深度权重  
    layer_weight = 1 - math.exp(-alpha * (k + 1) / L)  
    
    # 问题长度自适应因子  
    length_factor = (l_q / 30) ** 0.5  # 使用平方根来平滑长度影响  
    
    # 计算奇异值重要性  
    λ_importance = abs(Lambda[i]) / torch.norm(Lambda)  
    
    # 基础梯度影响  
    base_score = abs(Lambda[i] * torch.autograd.grad(loss, Lambda[i])[0])  
    
    # 带权重的P矩阵梯度影响  
    p_score = sum(  
        abs(P[j,i] * torch.autograd.grad(loss, P[j,i])[0]) *   
        math.exp(-j/d1)  # 距离衰减权重  
        for j in range(d1)  
    ) / d1  
    
    # 带权重的Q矩阵梯度影响  
    q_score = sum(  
        abs(Q[i,j] * torch.autograd.grad(loss, Q[i,j])[0]) *   
        math.exp(-j/d2)  # 距离衰减权重  
        for j in range(d2)  
    ) / d2  
    
    # 组合得分  
    S = layer_weight * length_factor * (  
        0.4 * λ_importance * base_score +   
        0.3 * p_score +   
        0.3 * q_score  
    )  
    
    return S  



def compute_satisfaction_score_v3(k, i, P, Q, λ, l_q, L, d1, d2, alpha):  
    """  
    方案2: Distance-Aware Gradient Impact Score  
    
    额外参数:  
    α: 层深度影响因子  
    """  
    # 层深度指数衰减  
    depth_factor = math.exp(alpha * k / L)  
    
    # 位置重要性权重（考虑与输入的距离）  
    position_weight = 1 / (1 + math.exp(-(k - L/2)))  # Sigmoid函数  
    
    # 问题长度归一化因子  
    length_norm = math.sqrt(l_q / 30)  # 30是基准长度  
    
    # 梯度敏感度计算（带距离衰减）  
    s_lambda = abs(λ[i] * λ[i].grad) if λ[i].grad is not None else 0  
    
    # P矩阵的梯度计算（考虑行位置的重要性）  
    if P[:, i].grad is not None:  
        p_grads = abs(P[:, i] * P[:, i].grad)  
        position_weights = torch.linspace(1, 0.5, d1)  # 线性衰减权重  
        s_P = torch.sum(p_grads * position_weights) / d1  
    else:  
        s_P = 0  
    
    # Q矩阵的梯度计算  
    s_Q = torch.sum(abs(Q[i, :] * Q[i, :].grad)) / d2 if Q[i, :].grad is not None else 0  
    
    # 组合梯度影响  
    gradient_impact = (s_lambda + s_P + s_Q) / 3  
    
    # 最终satisfaction score  
    score = depth_factor * position_weight * length_norm * gradient_impact  
    
    return score  






class FixedRankSVD:  
    
    '''
    实现每一层的suffix embedding的分解， 
    每一层的后缀矩阵分解为 P_k, Lambda_k, Q_k
    然后分别扩展为 P_k, Lambda_k, Q_k （使用Sparse Attention)
    
    
    '''
    def __init__(self, num_layers, hidden_size, min_steps, max_steps, config:BudgetSchedulerConfig):  
        self.num_layers = num_layers  
        self.hidden_size = hidden_size  
        self.min_steps = min_steps  # min(n_step)  
        self.max_steps = max_steps  # 2 * input_seq_length  
        self.config = config
        
    def initial_decompose(self, E_k):  
        """
        固定维度的SVD分解
        
        Args: 
            E_k：每一层的后缀矩阵，维度为[K_suffix, H]
        
        """ 
        # U, S, V 可以有两维， 也可以有3维 
        U, S, V = torch.svd(E_k)   # U-> P, S->\Lambda, V->Q
        # U.shape = [batch_size, K_suffix, K_suffix] 
        # S.shape = [batch_size, K_suffix, K_suffix]
        # V.shape = [batch_size, hidden_size, K_suffix]
        P_k = U[:, :self.min_steps]  
        Lambda_k = torch.diag(S[:self.min_steps])  # return a 1-D tensor
        Q_k = V[:, :self.min_steps]  
        return P_k, Lambda_k, Q_k
    
    def vertical_expansion(self, P_k, Lambda_k, Q_k, delta_r, satisfaction_score):  
        """
        垂直扩展SVD分解
        
        function：
            使用垂直扩展机制， 通过在P矩阵上方添加新的行来增加推理步骤
        """  
        current_steps = P_k.size(0)  
        new_steps = current_steps + delta_r  
        
        # 1. 生成新的行向量  
        P_new_rows = self.generate_new_rows(  
            P_k, Lambda_k, Q_k,   
            delta_r, satisfaction_score  
        )  
        
        # 2. 扩展P矩阵  
        P_extended = torch.cat([P_k, P_new_rows], dim=0)  
        
        # 3. 通过QR分解确保正交性  
        P_extended, R = torch.qr(P_extended)  
        
        # 4. 更新Lambda和Q  
        Lambda_extended = torch.mm(R, Lambda_k)  
        Q_extended = Q_k  
        
        return P_extended, Lambda_extended, Q_extended


    def generate_new_rows(self, P_k, Lambda_k, Q_k, delta_r, satisfaction_score):  
        """生成新的行向量"""  
        # 1. 计算注意力权重  
        attention_weights = self.compute_attention_weights(P_k, Lambda_k)  
        
        # 2. 生成新的行向量  
        new_rows = torch.zeros(delta_r, P_k.size(1))  
        
        for i in range(delta_r):  
            # 基于现有行的加权组合  
            weighted_sum = torch.mm(  
                attention_weights,   
                P_k  
            )  
            
            # 应用satisfaction score的影响  
            scale_factor = torch.sqrt(satisfaction_score)  
            new_row = scale_factor * weighted_sum  
            
            # 正交化  
            new_row = self.orthogonalize(new_row, P_k)  
            new_rows[i] = new_row  
            
        return new_rows  
    
    def increase_rank(self, P_k, Lambda_k, Q_k, satisfaction_score, delta_r):  
        """增加推理步骤数量"""  
        current_rank = Lambda_k.size(0)  
        new_rank = min(current_rank + delta_r, self.max_steps)  # delta_r 实际控制了推理步骤的增量
        
        # 计算新增奇异值和向量  
        new_singular_values = self.compute_new_values(  
            P_k, Lambda_k, Q_k, satisfaction_score  
        )  
        
        # 扩展矩阵维度  
        # P_new = self.extend_P(P_k, new_singular_values, delta_r)  
        # Lambda_new = self.extend_Lambda(Lambda_k, new_singular_values)  
        # Q_new = self.extend_Q(Q_k, new_singular_values, delta_r)  
        
        P_new, Lambda_new, Q_new  = self.extend_matrices(P_k, Lambda_k, Q_k, new_singular_values, delta_r)
        
        return P_new, Lambda_new, Q_new 
    
    def sparse_attention(self, scores, top_k=None):  
        n = scores.size(0)  
        weights = torch.zeros_like(scores)  
        
        # 对每一行进行处理  
        for i in range(n):  
            row = scores[i]  
            # 找到最大的top_k个值的索引  
            _, indices = torch.topk(row, min(top_k, n))  
            # 只在这些位置保留原始分数  
            weights[i, indices] = row[indices]  
        
        return weights 
    
    def compute_attention_weights(self, P_k, Lambda_k, top_k=None):  
        """  
        使用Sparse Attention计算注意力权重  
        
        参数:  
        P_k: 形状为[n, r]的矩阵，其中n是当前步数，r是秩  
        Lambda_k: 形状为[r, r]的对角矩阵，包含奇异值  
        top_k: 每个查询要关注的最大键值对数量  
        
        返回:  
        attention_weights: 形状为[n, n]的稀疏注意力权重矩阵  
        """  
        n, r = P_k.shape  
        
        # 如果没有指定top_k，设置一个默认值  
        if top_k is None:  
            top_k = min(int(math.sqrt(n)), n)  
        
        # 计算查询和键值矩阵  
        # 使用P_k和Lambda_k的组合作为特征表示  
        try:
            features = torch.mm(P_k, Lambda_k)  # [n, r] x [r, r] = [n, r]  
        except:
            try:
                raise RuntimeError(f"Invalid matrix shape for attention's matrix multiplication, \
                               \nwhere m1.shape = {P_k.shape}, m2.shape = {Lambda_k.shape}")
            except:
                raise TypeError(f"P_k.type = {type(P_k)}, Lambda_k.type = {type(Lambda_k)}, they do not has an attribute \'shape\'")
        # 计算注意力分数  
        attention_scores = torch.mm(features, features.t())  # [n, n]  
        attention_scores = attention_scores / math.sqrt(r)  # 缩放因子  
        
        # 实现Sparse Attention：只保留每行最大的top_k个值  
        # 应用稀疏化  
        sparse_scores = self.sparse_attention(attention_scores, top_k)  
        
        # 应用softmax得到最终的注意力权重  
        attention_weights = torch.nn.functional.softmax(sparse_scores, dim=-1)  
        
        return attention_weights  
    
    def compute_new_values(self, P_k, Lambda_k, Q_k, satisfaction_score):  
        """计算新增奇异值"""  
        # 使用Sparse Attention来生成新的奇异值  
        attention_weights = self.compute_attention_weights(P_k, Lambda_k)  # shape = [r, r]
        
        # 新奇异值计算公式  
        new_values = torch.sqrt(  
            satisfaction_score *   
            torch.mean(Lambda_k) *   
            attention_weights  
        )  
        
        return new_values
    
    def extend_matrices(self, P_k, Lambda_k, Q_k, new_values, delta_r):  
        """扩展SVD矩阵的具体实现"""  
        # 扩展P矩阵  
        P_new = torch.cat([  
            P_k,  
            self.compute_new_P_vectors(P_k, new_values, delta_r)  
        ], dim=1)  
        
        # 扩展Lambda矩阵  
        Lambda_new = torch.cat([  
            Lambda_k,  
            torch.diag(new_values)  
        ], dim=1)  
        
        # 扩展Q矩阵  
        Q_new = torch.cat([  
            Q_k,  
            self.compute_new_Q_vectors(Q_k, new_values, delta_r)  
        ], dim=1)  
        
        return P_new, Lambda_new, Q_new  

    def compute_new_P_vectors(self, P_k, new_values, delta_r):  
        """  
        计算新的P向量  
        
        参数:  
        P_k: 形状为[r, r]的矩阵  
        new_values: 新计算出来的奇异值  
        delta_r: 需要新增的向量数量  
        
        返回:  
        new_P: 形状为[delta_r, r]的新P向量矩阵  
        """  
        attention_weights  = self.compute_attention_weights(P_k, new_values) # shape = [r, r]
        
        
        # 初始化新的P向量矩阵  
        new_P = torch.zeros(delta_r, P_k.size(1))  # shape = [delta_r, r]
        
        for i in range(delta_r):
            # 使用注意力权重生成候选向量  
            weighted_sum = torch.sum(attention_weights, dim=0)  # [r]  
            candidate = weighted_sum.unsqueeze(0)  # [1, r]
            
            # 正交化处理  
            orthogonalized = self.orthogonalize(candidate, P_k)  
            new_P[i] = orthogonalized  
            
            # 更新P_k以便下一次生成  
            P_k = torch.cat([P_k, orthogonalized.unsqueeze(0)], dim=0) 
        
        # return self.orthogonalize(attention_weights , P_k)  
        return new_P
    

    def compute_new_Q_vectors(self, Q_k, new_values, delta_r):  
        """  
        计算新的Q向量  
        
        参数:  
        Q_k: 形状为[h, r]的矩阵，其中h是隐藏层维度  
        new_values: 新计算出来的奇异值矩阵  
        delta_r: 需要新增的向量数量  
        
        返回:  
        new_Q: 形状为[h, delta_r]的新Q向量矩阵  
        """  
        # Q_k.T的形状是[r, h]  
        # 计算注意力权重矩阵 [r, r]  
        attention_weights = self.compute_attention_weights(Q_k.T, new_values)  
        
        
        
        
        # 初始化新的Q向量矩阵 [h, delta_r]  
        new_Q = torch.zeros(Q_k.size(0), delta_r)  
        
        # 对每个需要生成的新向量  
        for i in range(delta_r):  
            # 使用注意力权重的加权和作为候选向量  
            candidate = torch.mm(Q_k, attention_weights)  # [h, r]  
            weighted_sum = torch.sum(candidate, dim=1)  # [h]  
            candidate = weighted_sum.unsqueeze(1)  # [h, 1]  
            
            # 正交化处理  
            orthogonalized = self.orthogonalize(candidate.T, Q_k.T).T  
            new_Q[:, i] = orthogonalized.squeeze()  
            
            # 更新Q_k以便下一次生成  
            Q_k = torch.cat([Q_k, orthogonalized], dim=1)  
        
        return new_Q  
    
    
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




class AdaptiveBudgetScheduler:  
    '''
    基于熵和动量的推理步数调度器
    
    '''
    def __init__(self, n_layers, min_steps, input_length, config:BudgetSchedulerConfig):  
        self.b_min = n_layers * min_steps  
        self.b_max = n_layers * 2 * input_length  
        self.momentum = 0  
        self.velocity = 0  
        self.t = 0  
        self.config = config  
        
        # 温度相关的状态变量  
        self.last_budget_grad = None  
        self.T0 = config.T0  
        self.last_budget = None 
        
    def compute_entropy(self, satisfaction_scores):  
        # 计算每层的熵 
        
        # 归一化满意度 
        probs = F.softmax(satisfaction_scores / self.config.tau, dim=-1)  
        # 计算层级熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  
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
        t = self.t / self.config.t_scale  
        return (self.config.A * torch.sin(self.config.omega * t + self.config.phi)**2   
                * torch.exp(-self.config.lambda_ * t))  
    
    def update_momentum(self, delta):  
        # 更新动量  
        self.momentum = (self.config.beta * self.momentum +   
                        (1 - self.config.beta) * delta)  
        self.velocity = (self.config.alpha * self.velocity +   
                        (1 - self.config.alpha) * self.momentum**2)  
        
    def __call__(self, satisfaction_scores):  
        # 计算熵  （层级熵）
        layer_entropy = self.compute_entropy(satisfaction_scores)  
        total_entropy = layer_entropy.mean()  
        
        # 计算探索项  
        explore = self.exploration_term()  
        
        # 计算基础预算  
        budget_ratio = torch.sigmoid(total_entropy / self.config.H_max)  
        
        # 预算 b^{(t)}
        base_budget = (self.b_min + (self.b_max - self.b_min) *   
                      (budget_ratio + explore))  
        
        # 更新动量  
        delta = base_budget - self.last_budget if self.t > 0 else 0  
        self.update_momentum(delta)  
        
        # 应用动量修正  
        final_budget = base_budget * (self.momentum /   
                                    (torch.sqrt(self.velocity) + 1e-8))  
        
        self.final_budget = final_budget
        # 计算温度  
        temp = self.compute_temperature()  
        
        # 分配层级预算  
        layer_budgets = self.allocate_layer_budgets(  
            final_budget, layer_entropy, temp)  
        
        self.last_budget = final_budget  
        self.t += 1  
        
        return layer_budgets
    
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
    
    
    def allocate_layer_budgets(self, final_budget, layer_entropy, temperature):  
        """分配层级预算  
        
        使用带温度的Softmax进行层级预算分配：  
        b_l^{(t)} = b^{(t)} · exp(H_l^{(t)}/T^{(t)}) / ∑ᵢexp(H_i^{(t)}/T^{(t)})  
        """  
        # 添加探索项的影响  
        explore = self.exploration_term()  
        adjusted_temp = temperature * (1 + self.config.gamma * explore)  
        
        # 计算分配权重  
        weights = torch.exp(layer_entropy / adjusted_temp)  
        normalized_weights = weights / weights.sum()  
        
        # 确保预算是整数  
        base_budgets = (normalized_weights * final_budget).floor()  
        
        # 处理舍入误差，确保总和等于final_budget  
        remaining = final_budget - base_budgets.sum()  
        if remaining > 0:  
            # 根据小数部分大小分配剩余预算  
            decimal_parts = normalized_weights * final_budget - base_budgets  
            _, indices = torch.sort(decimal_parts, descending=True)  
            for i in range(int(remaining)):  
                base_budgets[indices[i]] += 1  
        
        # 确保每层至少有最小预算  
        min_layer_budget = self.b_min / len(layer_entropy)  
        base_budgets = torch.maximum(base_budgets,   
                                    torch.tensor(min_layer_budget))  
        
        # 应用动态缩放以确保总预算约束  
        if base_budgets.sum() > final_budget:  
            scale = final_budget / base_budgets.sum()  
            base_budgets = (base_budgets * scale).floor()  
        
        return base_budgets 
    
    
    


class AdaptiveSingularValueEstimator:  
    def __init__(self, config: BudgetSchedulerConfig):  
        self.momentum = {}  # 每个奇异值的动量  
        self.velocity = {}  # 每个奇异值的速度  
        self.t = 0  
        self.config = config  
        
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
        计算探索项  
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
    
    
    def update_layer_singular_values(self, lambda_matrix, layer_idx, layer_budget, fixed_rank_svd=None):  
        """更新单层的奇异值  
        
        Args:  
            lambda_matrix: 当前层的奇异值向量 shape=(n_l,)  
            layer_idx: 层索引  
            layer_budget: 该层分配到的预算  
            fixed_rank_svd: FixedRankSVD对象，用于处理SVD分解相关的操作 
        Returns:  
            torch.Tensor: 更新后的奇异值向量  
        """  
        
        
        
        # 1. 首先通过FixedRankSVD扩展P、Lambda、Q矩阵  
        new_P = fixed_rank_svd.compute_new_P_vectors(layer_idx, layer_budget)
        new_Q = fixed_rank_svd.compute_new_Q_vectors(layer_idx, layer_budget)
        new_lambda = fixed_rank_svd.compute_new_Lambda_vectors(layer_idx, layer_budget)  
        
        # n_l = lambda_matrix.size(0)  # 该层的奇异值数量  
        # new_lambda = lambda_matrix.clone()  
        
        # 获取最大的layer_budget个奇异值的索引 
        
        # 这里有问题， 这里 min(layer_budget, n_l) 确保了每层被分配的预算不会超过该层原有的奇异值数量
        # 而我们之前的设计是拓展奇异值矩阵，所以这里新的的n_l直接等于layer_budget
        # 
        # _, top_indices = torch.topk(lambda_matrix, k=min(layer_budget, n_l))  
        
        # 使用扩展后的新维度更新奇异值   
        for i in range(layer_budget):  
            delta = self.compute_delta_lambda(new_lambda, i, layer_idx)  
            new_lambda[i] = new_lambda[i] + delta  
        
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
    # prompt_encoder = BidirectionalAttentionalPromptEncoder(**config)  
    
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