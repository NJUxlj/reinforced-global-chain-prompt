import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math  

class MultiHeadAttention(nn.Module):  
    """多头注意力机制"""  
    def __init__(self, d_model, num_heads):  
        super().__init__()  
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"  
        
        self.d_model = d_model  
        self.num_heads = num_heads  
        self.d_k = d_model // num_heads  
        
        # 定义线性变换层  
        self.W_q = nn.Linear(d_model, d_model)  
        self.W_k = nn.Linear(d_model, d_model)  
        self.W_v = nn.Linear(d_model, d_model)  
        self.W_o = nn.Linear(d_model, d_model)  
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):  
        """计算缩放点积注意力  
        
        Args:  
            Q: Query矩阵 [batch_size, num_heads, seq_len, d_k]  
            K: Key矩阵 [batch_size, num_heads, seq_len, d_k]  
            V: Value矩阵 [batch_size, num_heads, seq_len, d_k]  
            mask: 掩码张量 [batch_size, 1, seq_len, seq_len]  
        """  
        d_k = Q.size(-1)  
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  
        
        if mask is not None:  
            scores = scores.masked_fill(mask == 0, float('-inf'))  
            
        attention_weights = F.softmax(scores, dim=-1)  
        output = torch.matmul(attention_weights, V)  
        return output, attention_weights  
    
    def forward(self, Q, K, V, mask=None):  
        '''
        Q: [B, L, H]
        K: [B, L, H]
        V: [B, L, H]
        '''
        batch_size = Q.size(0)  
        
        # 线性变换  
        Q = self.W_q(Q)  # [B, L, H]
        K = self.W_k(K)  # [B, L, H]
        V = self.W_v(V)  
        
        # 重塑张量用于多头注意力  
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  
        
        # 计算注意力  
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)  
        
        # 重塑回原始维度  
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  
        output = self.W_o(output)  
        
        return output, attention_weights  

class PositionwiseFeedForward(nn.Module):  
    """位置前馈网络"""  
    def __init__(self, d_model, d_ff):  
        super().__init__()  
        self.fc1 = nn.Linear(d_model, d_ff)  
        self.fc2 = nn.Linear(d_ff, d_model)  
        self.relu = nn.ReLU()  
        
    def forward(self, x):  
        return self.fc2(self.relu(self.fc1(x)))  

class DecoderLayer(nn.Module):  
    """解码器层"""  
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  
        super().__init__()  
        self.self_attn = MultiHeadAttention(d_model, num_heads)  
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)  
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(dropout)  
        
    def forward(self, x:torch.Tensor, mask=None):  
        '''
        x.shape = [B, L, H]  # L: 序列长度, H: 隐藏层维度
        '''
        # 自注意力  
        attn_output, _ = self.self_attn(x, x, x, mask)  # [B, L, H]
        x = self.norm1(x + self.dropout(attn_output))  # [B, L, H]
        
        # 前馈网络  
        ff_output = self.feed_forward(x)  
        x = self.norm2(x + self.dropout(ff_output))  
        
        return x  

def generate_new_step(context: torch.Tensor,  
                     model: DecoderLayer = None,  
                     max_len: int = 64,  
                     temperature: float = 0.7) -> torch.Tensor:  
    """使用自定义Transformer生成新的推理步骤  
    
    Args:  
        context: 形状为 [seq_len, hidden_dim] 的上下文张量  
        model: 解码器模型  
        max_len: 生成的最大长度  
        temperature: 采样温度  
    
    Returns:  
        torch.Tensor: 形状为 [1, hidden_dim] 的新步骤张量  
    """  
    if model is None:  
        # 如果没有传入模型，创建一个新的解码器层  
        d_model = context.size(-1)  # hidden_dim  
        model = DecoderLayer(  
            d_model=d_model,  
            num_heads=8,  
            d_ff=d_model * 4,  
            dropout=0.1  
        )  
    
    device = context.device  
    model = model.to(device)  
    
    # 创建因果掩码  
    seq_len = context.size(0)  
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()  
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
    context = context.unsqueeze(0)  # [1, seq_len, hidden_dim]  
    
    with torch.no_grad():  
        # 通过解码器生成新的隐藏状态  
        output = model(context, causal_mask)  
        
        # 获取最后一个时间步的隐藏状态  
        last_hidden = output[:, -1:, :]  # [1, 1, hidden_dim]  
        
        # 应用温度缩放  
        if temperature != 1.0:  
            last_hidden = last_hidden / temperature  
        
        # 使用一个简单的线性层将隐藏状态映射到下一个token的logits  
        # projection = nn.Linear(last_hidden.size(-1), last_hidden.size(-1)).to(device)  
        # logits = projection(last_hidden)  
        
        # # 使用softmax获取概率分布  
        # probs = F.softmax(logits, dim=-1)  
        
        # 采样生成新的隐藏状态  
        # new_step = torch.multinomial(probs.squeeze(1), num_samples=1)  # [1, hidden_dim]  
    
    # return new_step.squeeze(0)  # [hidden_dim]  
    return last_hidden

# 使用示例  
if __name__ == "__main__":  
    # 创建示例数据  
    seq_len, hidden_dim = 10, 512  
    context = torch.randn(seq_len, hidden_dim)  
    
    # 创建解码器模型  
    decoder = DecoderLayer(  
        d_model=hidden_dim,  
        num_heads=8,  
        d_ff=hidden_dim * 4,  
        dropout=0.1  
    )  
    
    # 生成新步骤  
    new_step = generate_new_step(  
        context=context,  
        model=decoder,  
        temperature=0.7  
    )  
    
    print(f"生成的新步骤形状: {new_step.shape}")  # 应该是 [hidden_dim]