import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from simple_moe import TransformerMoE
from tokenizers import Tokenizer
import logging
from datetime import datetime

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model, train_loader, optimizer, epochs=3):
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.train()
    
    setup_logging(rank)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(rank)
            optimizer.zero_grad()
            output = model(batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 100 == 0 and rank == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    
    # 使用 tokenizers 库加载 tokenizer.json
    tokenizer = Tokenizer.from_file('data/tokenizer.json')
    
    dataset = JsonlDataset('data/distill_r1_110k_sft.jsonl', tokenizer)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    vocab_size = tokenizer.get_vocab_size()
    d_model = 768
    num_heads = 12
    num_layers = 6
    d_ff = 2048
    max_seq_len = 2048
    num_experts = 4
    k = 2
    
    # 初始化模型
    model = TransformerMoE(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_experts, k)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # 记录初始参数
    if torch.distributed.get_rank() == 0:
        logging.info("Initial Parameters:")
        logging.info(f"Vocab Size: {vocab_size}")
        logging.info(f"Model Dimensions: {d_model}")
        logging.info(f"Number of Heads: {num_heads}")
        logging.info(f"Number of Layers: {num_layers}")
        logging.info(f"Feedforward Dimensions: {d_ff}")
        logging.info(f"Max Sequence Length: {max_seq_len}")
        logging.info(f"Number of Experts: {num_experts}")
        logging.info(f"Top-k Experts: {k}")
        logging.info(f"Learning Rate: {1e-4}")
        logging.info(f"Batch Size: {8}")
        logging.info(f"Number of Epochs: {3}")
        logging.info(f"Number of GPUs: {world_size}")
    
    # 记录数据集指标
    if torch.distributed.get_rank() == 0:
        logging.info("Dataset Metrics:")
        logging.info(f"Number of Samples: {len(dataset)}")
        logging.info(f"Max Sequence Length: {max_seq_len}")
    
    # 启动分布式训练
    mp.spawn(train, args=(world_size, model, train_loader, optimizer), nprocs=world_size, join=True)
    
    # 记录训练结果
    if torch.distributed.get_rank() == 0:
        logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()