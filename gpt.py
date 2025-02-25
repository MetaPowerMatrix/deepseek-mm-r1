# gpt.py
import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Adding to sys.path:", project_root)
sys.path.append(project_root)

from model.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_length):
        super(GPT, self).__init__()
        self.hidden_dim = hidden_dim  # 添加 hidden_dim 变量



class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, embed_size * 4)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length)
        x = self.embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()  # 设置为评估模式
        
        # 获取初始输出
        generated_ids = input_ids
        for _ in range(max_length):
            outputs = self(generated_ids)
            logits = outputs  # 假设模型的输出是 logits
            logits = logits[:, -1, :]  # 只关注最新生成的 token

            # 应用温度采样
            logits = logits / temperature
            
            # Top-K 采样
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                top_k_probs = F.softmax(top_k_values, dim=-1)
                next_token = torch.multinomial(top_k_probs, 1)
                next_token = top_k_indices.gather(-1, next_token)
            else:
                # 默认采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # 添加生成的 token 到输入序列
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return generated_ids