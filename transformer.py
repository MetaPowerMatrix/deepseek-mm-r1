# transformer_block.py
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.q = nn.Linear(embed_size, head_size, bias=False)
        self.k = nn.Linear(embed_size, head_size, bias=False)
        self.v = nn.Linear(embed_size, head_size, bias=False)
        self.scale = head_size ** -0.5

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, v)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, feed_forward_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attention_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)