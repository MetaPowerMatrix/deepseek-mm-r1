import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
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
        seq_len = src.size(1)
        
        # 词嵌入和位置编码
        src = self.token_embedding(src) * torch.sqrt(torch.tensor(self.token_embedding.embedding_dim, dtype=torch.float32))
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, mask)
        
        # 处理每个位置的表示
        batch_size = src.size(0)
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
    # 创建一个简单的混合专家模型
    model = SimpleMoE(input_size=10, hidden_size=64, output_size=2, num_experts=8, k=2)
    
    # 生成随机输入数据
    batch_size = 16
    x = torch.randn(batch_size, 10)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"SimpleMoE 模型参数量: {count_parameters(model):.2f}M")
    
    print("\n=== 扩展模型到百万级参数 ===")
    
    # 创建一个更大的SimpleMoE模型
    large_model = SimpleMoE(input_size=1024, hidden_size=2048, output_size=1024, num_experts=32, k=4)
    large_x = torch.randn(batch_size, 1024)
    large_output = large_model(large_x)
    print(f"大型SimpleMoE - 输入形状: {large_x.shape}")
    print(f"大型SimpleMoE - 输出形状: {large_output.shape}")
    print(f"大型SimpleMoE 模型参数量: {count_parameters(large_model):.2f}M")
    
    # 测试TransformerMoE
    vocab_size = 50000  # 增大词汇表
    d_model = 1024      # 增大模型维度
    num_heads = 16
    num_layers = 6
    d_ff = 4096         # 增大前馈网络
    max_seq_len = 512
    num_experts = 16    # 增加专家数量
    
    # 创建模型
    transformer_moe = TransformerMoE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        num_experts=num_experts,
        k=2
    )
    
    # 生成随机输入
    batch_size = 16
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = transformer_moe(input_ids)
    print(f"大型Transformer MoE - 输入形状: {input_ids.shape}")
    print(f"大型Transformer MoE - 输出形状: {output.shape}")
    print(f"大型TransformerMoE 模型参数量: {count_parameters(transformer_moe):.2f}M")
    
    # 分析模型各部分参数量
    print("\n模型参数细节分析:")
    
    # TransformerMoE参数分析
    embedding_params = sum(p.numel() for p in transformer_moe.token_embedding.parameters()) / 1e6
    encoder_params = sum(p.numel() for layer in transformer_moe.encoder_layers 
                         for p in layer.parameters()) / 1e6
    moe_params = sum(p.numel() for p in transformer_moe.moe_layer.parameters()) / 1e6
    output_params = sum(p.numel() for p in transformer_moe.output_layer.parameters()) / 1e6
    
    print(f"  - 词嵌入层参数: {embedding_params:.2f}M ({embedding_params/count_parameters(transformer_moe)*100:.1f}%)")
    print(f"  - 编码器层参数: {encoder_params:.2f}M ({encoder_params/count_parameters(transformer_moe)*100:.1f}%)")
    print(f"  - MoE层参数: {moe_params:.2f}M ({moe_params/count_parameters(transformer_moe)*100:.1f}%)")
    print(f"  - 输出层参数: {output_params:.2f}M ({output_params/count_parameters(transformer_moe)*100:.1f}%)")
    
    # MoE层参数分析
    gate_params = sum(p.numel() for p in transformer_moe.moe_layer.gate.parameters()) / 1e6
    experts_params = sum(p.numel() for expert in transformer_moe.moe_layer.experts 
                          for p in expert.parameters()) / 1e6
    
    print(f"\nMoE层细节:")
    print(f"  - 门控网络参数: {gate_params:.2f}M ({gate_params/moe_params*100:.1f}%)")
    print(f"  - 专家网络参数: {experts_params:.2f}M ({experts_params/moe_params*100:.1f}%)")
    
    # 每个专家平均参数量
    avg_expert_params = experts_params / transformer_moe.moe_layer.num_experts
    print(f"  - 每个专家平均参数: {avg_expert_params:.2f}M")
    
    # 计算与传统Transformer对比
    # 假设传统模型用相同参数量的FFN替代MoE
    traditional_ffn_params = 2 * d_model * d_ff / 1e6  # 两个线性层
    print(f"\n比较:")
    print(f"  - MoE层总参数: {moe_params:.2f}M")
    print(f"  - 等价传统FFN层参数: {traditional_ffn_params:.2f}M")
    print(f"  - 参数量比例: {moe_params/traditional_ffn_params:.1f}x")
    
    # 总结
    print("\n模型规模总结:")
    print(f"小型SimpleMoE: {count_parameters(model):.2f}M 参数")
    print(f"大型SimpleMoE: {count_parameters(large_model):.2f}M 参数")
    print(f"大型TransformerMoE: {count_parameters(transformer_moe):.2f}M 参数")