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






class FixedRankSVDReasoningAllocator:  
    def __init__(self, num_layers, hidden_size, min_steps, max_steps):  
        self.num_layers = num_layers  
        self.hidden_size = hidden_size  
        self.min_steps = min_steps  # min(n_step)  
        self.max_steps = max_steps  # 2 * input_seq_length  
        
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
        features = torch.mm(P_k, Lambda_k)  # [n, r] x [r, r] = [n, r]  
        
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
        attention_weights = self.compute_attention_weights(P_k, Lambda_k)  
        
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
        new_Q: 形状为[delta_r, h]的新Q向量矩阵  
        """  
        # Q_k.T的形状是[r, h]  
        # 计算注意力权重矩阵 
        attention_weights = self.compute_attention_weights(Q_k, new_values)  # shape = [h, h]
        
        # 初始化新的Q向量矩阵 [delta_r, h]  
        new_Q = torch.zeros(delta_r, Q_k.size(1))  
        
        # 对每个需要生成的新向量  
        for i in range(delta_r):  
            # 使用注意力权重的加权和作为候选向量  
            candidate = torch.mm(attention_weights, Q_k)  # [r, h]  
            weighted_sum = torch.sum(candidate, dim=0)  # [h]  
            candidate = weighted_sum.unsqueeze(0)  # [1, h]  
            
            # 正交化处理  
            orthogonalized = self.orthogonalize(candidate, Q_k)  
            new_Q[i] = orthogonalized  
            
            # 更新Q_k以便下一次生成  
            Q_k = torch.cat([Q_k, orthogonalized.unsqueeze(0)], dim=0)  
        
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