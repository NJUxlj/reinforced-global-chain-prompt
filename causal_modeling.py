import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math  

from transformers import (
    BertModel
)

from config import Config

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
        
        print("Q.shape = ",Q.shape)
        print("K.shape = ",K.shape)
        print("V.shape = ",V.shape)
        
        
        # 计算注意力  
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)  
        
        # 重塑回原始维度  
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  
        output = self.W_o(output)  
        print("attention_output.shape = ", output.shape)
        
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
        
        print("feedforward_output.shape = ", x.shape)
        
        return x  

class RollbackDecoderWithHead(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_ff, 
        num_heads = 8, 
        dropout=0.1,
        num_layers=2,
        ):
        super().__init__()
        self.decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.head = nn.Linear(d_model, d_model)
        self.bert_model = BertModel.from_pretrained(Config["models"]["bert-base-uncased"]["model_path"] )

        self.token_embedding = self.bert_model.embeddings.word_embeddings  # shape = [vocab_size, d_model] 
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([
            self.decoder_layer
            for _ in range(self.num_layers)
        ])
        
        self.vocab_size = self.bert_model.config.vocab_size
        # 输出投影到词表大小  
        self.classifier = nn.Linear(d_model, self.vocab_size)  
        
        
    def forward(self, x, mask=None):  
        decoder_output = x
        for i in range(self.num_layers):
            decoder_output =self.decoders[i](decoder_output) # [B, L, H]
        
        hidden_state = decoder_output[:, -1, :]  # [B, H]
        logits = self.classifier(hidden_state)  # [B, vocab_size]
        return logits  
    
    def embed_tokens(self, token_ids):  
        """将token ID转换为embeddings"""  
        return self.token_embedding(token_ids)
        
        
        
def generate_new_step(context: torch.Tensor,  
                     model: RollbackDecoderWithHead = None,  
                     temperature: float = 0.7) -> torch.Tensor:  
    """生成下一个token并返回其embedding  
    
    Args:  
        context: 形状为 [seq_len, hidden_dim] 的上下文张量  
        model: 带有输出头的解码器模型  
        temperature: 采样温度  
        vocab_size: 词表大小（默认使用BERT的词表大小）  
    
    Returns:  
        tuple: (next_token_id, next_token_embedding)  
            - next_token_id: 下一个token的ID  
            - next_token_embedding: 下一个token的embedding向量 [hidden_dim]  
    """  
    if model is None:  
        # 如果没有传入模型，创建一个新的解码器层  
        d_model = context.size(-1)  # hidden_dim  
        model = RollbackDecoderWithHead(  
            d_model=d_model,  
            d_ff=d_model * 4, 
            num_heads=8,  
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
        next_token_logits  = model(context, causal_mask)  # shape = [1, vocab_size]
        
        
        # 应用温度缩放  
        # # temperature 影响概率分布的"锐利度", <1 使分布更尖锐，>1 使分布更平缓    
        if temperature != 1.0:  
            next_token_logits = next_token_logits / temperature  
        
        # 计算概率分布  
        probs = F.softmax(next_token_logits, dim=-1)  
        
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
        next_token_embedding = model.embed_tokens(next_token)  # [1, 1, hidden_dim]  
        next_token_embedding = next_token_embedding.squeeze()  # [hidden_dim]  
        

    # return new_step.squeeze(0)  # [hidden_dim]  
    return next_token.squeeze().item(), next_token_embedding

# 使用示例  
if __name__ == "__main__":  
    # 创建示例数据  
    seq_len, hidden_dim = 10, 512  
    context = torch.randn(seq_len, hidden_dim)  
    
    # 创建解码器模型  
    # decoder = DecoderLayer(  
    #     d_model=hidden_dim,  
    #     num_heads=8,  
    #     d_ff=hidden_dim * 4,  
    #     dropout=0.1  
    # )  
    
    # 生成新步骤  
    _,next_token_embedding = generate_new_step(  
        context=context,  
        model=None,  
        temperature=0.7  
    )  
    
    print(f"生成的新步骤形状: {next_token_embedding.shape}")  # 应该是 [hidden_dim]