import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embeddings import RotaryEmbedding, YarnRotaryEmbedding, apply_rotary_pos_emb
import os
import json

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=q.device))
        
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
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6, d_ff=2048, max_seq_len=2048, num_experts=4, k=2, dropout=0.1, pad_token_id=None):
        super(TransformerMoE, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.k = k
        self.pad_token_id = pad_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
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
    
    def forward(self, src=None, mask=None, input_ids=None, attention_mask=None, labels=None):
        """
        前向传播函数，支持两种调用方式：
        1. 直接传入src和mask
        2. 使用HuggingFace风格的参数：input_ids, attention_mask, labels
        """
        # 如果提供了input_ids，将其作为src
        if input_ids is not None:
            src = input_ids
            
        # 如果提供了attention_mask，将其作为mask
        if attention_mask is not None:
            # 转换attention_mask格式为多头注意力的格式
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # 确保有有效的src
        if src is None:
            raise ValueError("必须提供input_ids或src参数")
        
        # src: [batch_size, seq_len]
        batch_size, seq_len = src.size()
        
        # 安全检查：确保输入索引不超过词表大小
        src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # 如果未提供掩码且有pad_token_id，则创建注意力掩码
        if mask is None and self.pad_token_id is not None:
            # 创建attention mask，将pad_token_id标记为0，其他为1
            mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # 词嵌入
        embedding_dim_tensor = torch.tensor(self.token_embedding.embedding_dim, 
                                           dtype=torch.float32, 
                                           device=src.device)
        src = self.token_embedding(src) * torch.sqrt(embedding_dim_tensor)
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
        logits = self.output_layer(output)
        
        # 如果提供了标签，计算损失
        loss = None
        if labels is not None:
            # 使用交叉熵损失计算
            # 重塑为 [batch_size * seq_len, vocab_size]
            logits_view = logits.view(-1, self.vocab_size)
            # 重塑为 [batch_size * seq_len]
            labels_view = labels.view(-1)
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits_view, labels_view)
        
        # 返回结果，支持HuggingFace风格的返回格式
        if loss is not None:
            return type('ModelOutput', (), {'loss': loss, 'logits': logits})
        return logits
    
    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """
        从预训练模型加载权重
        
        参数:
            model_path: 预训练模型路径，可以是本地文件路径或Hugging Face模型ID
            device: 设备，如'cuda'或'cpu'
            
        返回:
            加载了预训练权重的TransformerMoE模型
        """
        # 加载模型配置和权重
        if os.path.isdir(model_path):
            # 本地文件路径
            config_path = os.path.join(model_path, "config.json")
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            
            # 检查文件是否存在
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"权重文件不存在: {weights_path}")
            
            # 加载配置
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 加载权重
            weights = torch.load(weights_path, map_location=torch.device('cpu'))
        else:
            # 尝试从Hugging Face加载
            try:
                from transformers import AutoConfig, AutoModel
                
                # 加载配置
                config = AutoConfig.from_pretrained(model_path).to_dict()
                
                # 加载预训练模型
                hf_model = AutoModel.from_pretrained(model_path)
                
                # 提取权重
                weights = {k: v for k, v in hf_model.state_dict().items()}
                
                del hf_model  # 释放内存
            except ImportError:
                raise ImportError("使用Hugging Face模型需要安装transformers库")
            except Exception as e:
                raise ValueError(f"从Hugging Face加载模型失败: {str(e)}")
        
        # 创建模型实例
        model = cls(
            vocab_size=config.get("vocab_size"),
            d_model=config.get("d_model", config.get("hidden_size", 768)),
            num_heads=config.get("num_heads", config.get("num_attention_heads", 12)),
            num_layers=config.get("num_layers", config.get("num_hidden_layers", 6)),
            d_ff=config.get("d_ff", config.get("intermediate_size", 2048)),
            max_seq_len=config.get("max_seq_len", config.get("max_position_embeddings", 2048)),
            num_experts=config.get("num_experts", 4),
            k=config.get("k", 2),
            dropout=config.get("dropout", config.get("hidden_dropout_prob", 0.1)),
            pad_token_id=config.get("pad_token_id")
        )
        
        # 加载权重到模型
        # 处理权重名称不匹配的情况
        model_state_dict = model.state_dict()
        new_weights = {}
        
        for name, tensor in weights.items():
            # 尝试将Hugging Face模型的权重映射到我们的模型
            if name in model_state_dict:
                if tensor.shape == model_state_dict[name].shape:
                    new_weights[name] = tensor
                else:
                    print(f"警告: 权重形状不匹配: {name}, 预期 {model_state_dict[name].shape}, 实际 {tensor.shape}")
            else:
                # 尝试映射常见的命名差异
                mapped_name = None
                
                # 词嵌入映射
                if "embeddings.word_embeddings.weight" in name:
                    mapped_name = "token_embedding.weight"
                # 层归一化映射
                elif "LayerNorm" in name:
                    if "encoder.layer" in name:
                        layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                        if layer_idx < model.num_layers - 1:
                            if "attention" in name and "LayerNorm" in name:
                                mapped_name = f"encoder_layers.{layer_idx}.norm1.weight"
                            elif "output" in name and "LayerNorm" in name:
                                mapped_name = f"encoder_layers.{layer_idx}.norm2.weight"
                    elif "encoder.output.LayerNorm" in name:
                        mapped_name = "final_layer_norm.weight"
                
                if mapped_name and mapped_name in model_state_dict:
                    if tensor.shape == model_state_dict[mapped_name].shape:
                        new_weights[mapped_name] = tensor
                        print(f"映射权重: {name} -> {mapped_name}")
                    else:
                        print(f"警告: 映射权重形状不匹配: {name} -> {mapped_name}")
        
        # 加载匹配的权重
        missing, unexpected = model.load_state_dict(new_weights, strict=False)
        
        if missing:
            print(f"缺少的权重: {len(missing)} 项")
            if len(missing) < 10:  # 只显示前几个
                print(missing)
        
        if unexpected:
            print(f"意外的权重: {len(unexpected)} 项")
            if len(unexpected) < 10:  # 只显示前几个
                print(unexpected)
        
        # 将模型移动到指定设备
        if device:
            model = model.to(device)
        
        # 设置为评估模式
        model.eval()
        
        return model
    
    def save_pretrained(self, save_directory):
        """
        保存模型到指定目录
        
        参数:
            save_directory: 保存模型的目录
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "k": self.k,
            "model_type": "transformer_moe",
            "pad_token_id": self.pad_token_id
        }
        
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        print(f"模型已保存到: {save_directory}")

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
    pad_token_id = 0  # 默认使用0作为padding token
    
    transformer_moe = TransformerMoE(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        d_ff=d_ff, 
        max_seq_len=max_seq_len, 
        num_experts=num_experts, 
        k=k,
        pad_token_id=pad_token_id
    )
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 设置一些padding值来测试掩码功能
    src[:, -50:] = pad_token_id
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
        scaling_factor=8.0,  # 8倍扩展
        rope_theta=10000,  # 10k作为rope_theta
        pad_token_id=pad_token_id
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
                 num_experts=4, k=2, dropout=0.1, scaling_factor=8.0, rope_theta=10000, pad_token_id=None):
        super(LongContextTransformerMoE, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
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
    
    def forward(self, src=None, mask=None, input_ids=None, attention_mask=None, labels=None):
        """
        前向传播函数，支持两种调用方式：
        1. 直接传入src和mask
        2. 使用HuggingFace风格的参数：input_ids, attention_mask, labels
        """
        # 如果提供了input_ids，将其作为src
        if input_ids is not None:
            src = input_ids
            
        # 如果提供了attention_mask，将其作为mask
        if attention_mask is not None:
            # 转换attention_mask格式为多头注意力的格式
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # 确保有有效的src
        if src is None:
            raise ValueError("必须提供input_ids或src参数")
        
        # src: [batch_size, seq_len]
        batch_size, seq_len = src.size()
        
        # 安全检查：确保输入索引不超过词表大小
        src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # 如果未提供掩码且有pad_token_id，则创建注意力掩码
        if mask is None and self.pad_token_id is not None:
            # 创建attention mask，将pad_token_id标记为0，其他为1
            mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # 词嵌入
        embedding_dim_tensor = torch.tensor(self.token_embedding.embedding_dim, 
                                           dtype=torch.float32, 
                                           device=src.device)
        src = self.token_embedding(src) * torch.sqrt(embedding_dim_tensor)
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
        logits = self.output_layer(output)
        
        # 如果提供了标签，计算损失
        loss = None
        if labels is not None:
            # 使用交叉熵损失计算
            # 重塑为 [batch_size * seq_len, vocab_size]
            logits_view = logits.view(-1, self.vocab_size)
            # 重塑为 [batch_size * seq_len]
            labels_view = labels.view(-1)
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits_view, labels_view)
        
        # 返回结果，支持HuggingFace风格的返回格式
        if loss is not None:
            return type('ModelOutput', (), {'loss': loss, 'logits': logits})
        return logits

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
        
        # 应用旋转位置编码
        if rotary_emb is not None:
            seq_len = q.size(2)
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
            cos, sin = rotary_emb(q, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 计算注意力分数 [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=q.device))
        
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