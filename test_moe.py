"""
测试MoE模型的脚本
"""

import torch
import os
import logging
from simple_moe import SimpleMoE, TransformerMoE, LongContextTransformerMoE

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_moe_models():
    # 测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 测试SimplesMoE
    input_size = 1024
    hidden_size = 2048
    output_size = 1024
    batch_size = 2
    
    # 创建模型
    logging.info("测试SimpleMoE模型...")
    model = SimpleMoE(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_experts=4,
        k=2
    ).to(device)
    
    # 创建随机输入
    x = torch.randn(batch_size, input_size).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    logging.info(f"输入形状: {x.shape}, 输出形状: {output.shape}")
    
    # 测试TransformerMoE
    logging.info("\n测试TransformerMoE模型...")
    vocab_size = 30000
    d_model = 768
    seq_len = 10
    
    # 创建模型
    transformer = TransformerMoE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        d_ff=1024,
        max_seq_len=512,
        num_experts=4,
        k=2
    ).to(device)
    
    # 创建随机输入
    src = torch.randint(0, vocab_size-1, (batch_size, seq_len)).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = transformer(src)
    
    logging.info(f"输入形状: {src.shape}, 输出形状: {output.shape}")
    
if __name__ == "__main__":
    test_moe_models() 