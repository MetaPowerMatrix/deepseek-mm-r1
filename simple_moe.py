import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embeddings import RotaryEmbedding, YarnRotaryEmbedding, apply_rotary_pos_emb

class MoELayer(nn.Module):
    """简单的混合专家模型层"""
    
    def __init__(self, input_size, output_size, num_experts=4, k=2):
        super(MoELayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k  # 每次使用的专家数量
        
        # 门控网络，用于选择专家
        self.gate = nn.Linear(input_size, num_experts)
        
        # 多个专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 4 * input_size),
                nn.ReLU(),
                nn.Linear(4 * input_size, output_size)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        前向传播
        x: [batch_size, input_size]
        """
        batch_size = x.size(0)
        
        # 计算门控分数
        router_logits = self.gate(x)  # [batch_size, num_experts]
        
        # 使用top-k选择专家
        routing_weights, indices = torch.topk(router_logits, self.k, dim=1)  # 选择得分最高的k个专家
        routing_weights = F.softmax(routing_weights, dim=1)  # 对选中的专家进行归一化
        
        # 为每个样本收集所有选中专家的输出
        expert_outputs = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # 对每个专家分别计算结果
        for i in range(self.k):
            # 获取当前专家索引
            expert_idx = indices[:, i]  # [batch_size]
            
            # 获取对应的权重
            weight = routing_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # 对每个样本使用选择的专家
            for j in range(batch_size):
                expert_out = self.experts[expert_idx[j]](x[j:j+1])  # [1, output_size]
                expert_outputs[j:j+1] += weight[j] * expert_out
        
        return expert_outputs

class SimpleMoE(nn.Module):
    """简单的混合专家模型"""
    
    def __init__(self, input_size, hidden_size, output_size, num_experts=4, k=2):
        super(SimpleMoE, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.moe_layer = MoELayer(hidden_size, hidden_size, num_experts, k)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.moe_layer(x)
        x = self.output_layer(x)
        return x

# Transformer相关组件
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q)  # [batch_size, seq_len, d_model]
        k = self.k_linear(k)  # [batch_size, seq_len, d_model]
        v = self.v_linear(v)  # [batch_size, seq_len, d_model]
        
        # 分割多头 [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 应用旋转位置编码（RoPE）
        seq_len = q.size(2)
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
        
        # 生成旋转编码
        cos, sin = RotaryEmbedding(self.head_dim)(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 计算注意力分数 [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重并应用
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑并拼接 [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.out_linear(attn_output)
        
        return output

class PositionWiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerMoE(nn.Module):
    """基于Transformer的混合专家模型"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_experts=4, k=2, dropout=0.1):
        super(TransformerMoE, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        
        # 使用RoPE替代原来的位置编码
        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_seq_len)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers-1)  # 保留一层用于MoE
        ])
        
        # MoE层
        self.moe_layer = MoELayer(d_model, d_model, num_experts, k)
        
        # 最终的编码器层
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        # src: [batch_size, seq_len]
        batch_size, seq_len = src.size()
        
        # 安全检查：确保输入索引不超过词表大小
        src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # 词嵌入
        src = self.token_embedding(src) * torch.sqrt(torch.tensor(self.token_embedding.embedding_dim, dtype=torch.float32))
        src = self.dropout(src)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, mask)
        
        # 处理每个位置的表示
        # 重塑为 [batch_size * seq_len, d_model]
        reshaped_src = src.view(batch_size * seq_len, -1)
        
        # 通过MoE层
        moe_output = self.moe_layer(reshaped_src)
        
        # 重塑回 [batch_size, seq_len, d_model]
        output = moe_output.view(batch_size, seq_len, -1)
        
        # 最终层归一化
        output = self.final_layer_norm(output)
        
        # 输出层
        output = self.output_layer(output)
        
        return output

# 计算模型参数数量
def count_parameters(model):
    """计算模型的参数数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# 使用示例
if __name__ == "__main__":
    # 测试代码
    input_size = 1024
    hidden_size = 1024
    output_size = 1024
    num_experts = 4
    k = 2
    
    # 测试MoE层
    moe_layer = MoELayer(input_size, output_size, num_experts, k)
    x = torch.randn(32, input_size)
    output = moe_layer(x)
    print(f"MoE层输出形状: {output.shape}")
    
    # 测试SimpleMoE模型
    simple_moe = SimpleMoE(input_size, hidden_size, output_size, num_experts, k)
    output = simple_moe(x)
    print(f"SimpleMoE输出形状: {output.shape}")
    
    # 测试TransformerMoE模型
    vocab_size = 30000
    d_model = 768
    num_heads = 12
    num_layers = 6
    d_ff = 2048
    max_seq_len = 2048
    batch_size = 8
    seq_len = 512
    
    transformer_moe = TransformerMoE(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_experts, k)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = transformer_moe(src)
    print(f"TransformerMoE输出形状: {output.shape}")
    
    # 测试LongContextTransformerMoE模型，支持超长序列
    long_seq_len = 8192  # 8k长度序列
    long_moe = LongContextTransformerMoE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=16384,  # 支持16k上下文长度
        num_experts=num_experts,
        k=k,
        scaling_factor=8.0  # 8倍扩展
    )
    
    # 在较短序列上测试
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = long_moe(src)
    print(f"LongContextTransformerMoE (短序列)输出形状: {output.shape}")
    
    # 尝试在长序列上测试（如果内存允许）
    try:
        long_src = torch.randint(0, vocab_size, (4, long_seq_len))  # 减小batch size以节省内存
        long_output = long_moe(long_src)
        print(f"LongContextTransformerMoE (长序列 {long_seq_len})输出形状: {long_output.shape}")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU内存不足，无法处理{long_seq_len}长度的序列。但模型理论上支持该长度。")
        else:
            print(f"运行错误: {e}")
    
    # 比较各模型参数数量
    print("\n模型参数统计:")
    print(f"MoE层: {count_parameters(moe_layer):.2f}M 参数")
    print(f"SimpleMoE: {count_parameters(simple_moe):.2f}M 参数")
    print(f"TransformerMoE: {count_parameters(transformer_moe):.2f}M 参数")
    print(f"LongContextTransformerMoE: {count_parameters(long_moe):.2f}M 参数")

class LongContextTransformerMoE(nn.Module):
    """支持长序列的Transformer混合专家模型"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, 
                 num_experts=4, k=2, dropout=0.1, scaling_factor=8.0, rope_theta=10000):
        super(LongContextTransformerMoE, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        
        # 使用YaRN RoPE实现长序列支持
        head_dim = d_model // num_heads
        self.rotary_emb = YarnRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=2048,  # 假设原始训练长度为2048
            beta_fast=32,
            beta_slow=1,
            mscale=1,
        )
        
        # 编码器层，使用支持长序列的注意力机制
        self.encoder_layers = nn.ModuleList([
            LongContextTransformerEncoderLayer(d_model, num_heads, d_ff, dropout, head_dim)
            for _ in range(num_layers-1)  # 保留一层用于MoE
        ])
        
        # MoE层
        self.moe_layer = MoELayer(d_model, d_model, num_experts, k)
        
        # 最终的编码器层
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        # src: [batch_size, seq_len]
        batch_size, seq_len = src.size()
        
        # 安全检查：确保输入索引不超过词表大小
        src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # 词嵌入
        src = self.token_embedding(src) * torch.sqrt(torch.tensor(self.token_embedding.embedding_dim, dtype=torch.float32))
        src = self.dropout(src)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, mask, self.rotary_emb)
        
        # 处理每个位置的表示
        # 重塑为 [batch_size * seq_len, d_model]
        reshaped_src = src.view(batch_size * seq_len, -1)
        
        # 通过MoE层
        moe_output = self.moe_layer(reshaped_src)
        
        # 重塑回 [batch_size, seq_len, d_model]
        output = moe_output.view(batch_size, seq_len, -1)
        
        # 最终层归一化
        output = self.final_layer_norm(output)
        
        # 输出层
        output = self.output_layer(output)
        
        return output

class LongContextMultiHeadAttention(nn.Module):
    """支持长序列的多头注意力机制"""
    
    def __init__(self, d_model, num_heads, head_dim=None):
        super(LongContextMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads if head_dim is None else head_dim
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None, rotary_emb=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q)  # [batch_size, seq_len, d_model]
        k = self.k_linear(k)  # [batch_size, seq_len, d_model]
        v = self.v_linear(v)  # [batch_size, seq_len, d_model]
        
        # 分割多头 [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 应用旋转位置编码（RoPE）
        if rotary_emb is not None:
            seq_len = q.size(2)
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
            
            # 生成旋转编码
            cos, sin = rotary_emb(q, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 计算注意力分数 [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重并应用
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑并拼接 [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.out_linear(attn_output)
        
        return output

class LongContextTransformerEncoderLayer(nn.Module):
    """支持长序列的Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, head_dim=None):
        super(LongContextTransformerEncoderLayer, self).__init__()
        self.self_attn = LongContextMultiHeadAttention(d_model, num_heads, head_dim)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, rotary_emb=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask, rotary_emb)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x