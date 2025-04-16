import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from simple_moe import TransformerMoE
from tokenizers import Tokenizer
import logging
import deepspeed
from deepspeed.ops.adam import FusedAdam


# 配置日志
def setup_logging(rank):
    log_format = f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(line.strip())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids[:self.max_seq_len]  # 截断到最大长度
        tokens += [0] * (self.max_seq_len - len(tokens))  # 填充到最大长度
        return torch.tensor(tokens)

def main():
    world_size = torch.cuda.device_count()
    
    # 配置日志
    setup_logging(0)
    
    # 记录数据加载开始
    logging.info("Loading dataset...")
    
    # 使用 tokenizers 库加载 tokenizer.json
    tokenizer = Tokenizer.from_file('data/tokenizer.json')
    
    dataset = JsonlDataset('data/distill_r1_110k_sft.jsonl', tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # 与 micro_batch_size_per_gpu 一致
    
    # 记录数据加载完成
    logging.info("Dataset loaded successfully.")
    
    vocab_size = tokenizer.get_vocab_size()
    d_model = 768
    num_heads = 12
    num_layers = 6
    d_ff = 2048
    max_seq_len = 2048
    num_experts = 4
    k = 2
    
    # 记录模型初始化开始
    logging.info("Initializing model...")
    
    # 初始化模型
    model = TransformerMoE(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_experts, k)
    # 使用 FusedAdam 优化器
    optimizer = FusedAdam(model.parameters(), lr=1e-4)

    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 24,  # 全局批量大小
        "train_micro_batch_size_per_gpu": 4,  # 每个 GPU 的微批次大小
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,  # 使用 ZeRO 优化
            "offload_optimizer": {
                "device": "cpu",  # 将优化器状态 Offload 到 CPU
            },
            "offload_param": {
                "device": "cpu",  # 将模型参数 Offload 到 CPU
            }
        }
    }
    
    # 使用 DeepSpeed 初始化
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )
    
    # 记录模型初始化完成
    logging.info("Model initialized successfully.")
    
    # 记录训练启动
    logging.info("Starting distributed training...")
    
    # 启动分布式训练
    for epoch in range(3):
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(model.device)
            optimizer.zero_grad()
            output = model(batch)
            loss = output.loss
            model.backward(loss)
            model.step()
            epoch_loss += loss.item()
            
            if i % 100 == 0 and rank == 0:
                logging.info(f"Epoch {epoch+1}/{3}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{3} completed. Average Loss: {avg_epoch_loss:.4f}")
    
    # 记录训练完成
    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()