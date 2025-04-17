import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from simple_moe import TransformerMoE
from tokenizers import Tokenizer
import logging
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim.oss import OSS
from fairscale.nn import checkpoint_wrapper
from torch.cuda.amp import autocast, GradScaler
import datetime

# 配置 NCCL 环境变量
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

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
    # 使用 TCP 初始化
    init_method = f"tcp://localhost:12356"
    try:
        # 使用 Gloo 后端
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1200))
        logging.info(f"Rank {rank}: Process group initialized successfully.")
    except Exception as e:
        logging.error(f"Rank {rank}: Process group initialization failed: {e}")
        raise

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, model, train_loader, optimizer, epochs=3):
    # 使用梯度缩放器
    scaler = GradScaler()
    
    # 记录初始参数
    if rank == 0:
        logging.info("Initial Parameters:")
        logging.info(f"Vocab Size: {model.module.vocab_size}")
        logging.info(f"Model Dimensions: {model.module.d_model}")
        logging.info(f"Number of Heads: {model.module.num_heads}")
        logging.info(f"Number of Layers: {model.module.num_layers}")
        logging.info(f"Feedforward Dimensions: {model.module.d_ff}")
        logging.info(f"Max Sequence Length: {model.module.max_seq_len}")
        logging.info(f"Number of Experts: {model.module.num_experts}")
        logging.info(f"Top-k Experts: {model.module.k}")
        logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"Batch Size: {train_loader.batch_size}")
        logging.info(f"Number of Epochs: {epochs}")
        logging.info(f"Number of GPUs: {world_size}")
    
    # 记录数据集指标
    if rank == 0:
        logging.info("Dataset Metrics:")
        logging.info(f"Number of Samples: {len(train_loader.dataset)}")
        logging.info(f"Max Sequence Length: {model.module.max_seq_len}")
    
    # 设置梯度累积步数
    accumulation_steps = 4
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(rank)
            
            # 使用混合精度训练
            with autocast():
                output = model(batch)
                loss = output.loss / accumulation_steps  # 缩放损失
            
            # 使用梯度缩放器
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            
            if i % 100 == 0 and rank == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item() * accumulation_steps:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    # 配置日志
    setup_logging(local_rank)
    
    # 记录数据加载开始
    if local_rank == 0:
        logging.info("Loading dataset...")
    
    # 使用 tokenizers 库加载 tokenizer.json
    tokenizer = Tokenizer.from_file('data/tokenizer.json')
    
    dataset = JsonlDataset('data/distill_r1_110k_sft.jsonl', tokenizer)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 进一步减少批量大小
    
    # 记录数据加载完成
    if local_rank == 0:
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
    if local_rank == 0:
        logging.info("Initializing model...")
    
    # 初始化模型
    model = TransformerMoE(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, num_experts, k)
    
    # 记录模型初始化完成
    if local_rank == 0:
        logging.info("Model initialized successfully.")
        
    # 在每个进程中初始化分布式环境
    setup(local_rank, world_size)
    
    model = model.to(local_rank)
    
    # 使用激活值检查点
    model = checkpoint_wrapper(model)
    
    # 基础优化器的参数
    base_optimizer_args = {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "eps": 1e-8
    }
    
    # 使用 FairScale 的 OSS 优化器
    optimizer = OSS(
        params=model.parameters(),
        optim=AdamW,  # 基础优化器类
        cpu_offload=True,  # OSS 的参数
        broadcast_fp16=True,  # OSS 的参数
        **base_optimizer_args  # 传递给基础优化器的参数
    )
    
    # 使用 FairScale 的 ShardedDataParallel
    model = ShardedDataParallel(model, optimizer)
    model.train()
    
    # 记录训练启动
    if local_rank == 0:
        logging.info("Starting distributed training...")

    # 训练模型
    train_model(local_rank, world_size, model, train_loader, optimizer)
    
    # 在每个进程中清理分布式环境
    cleanup()
    
    # 记录训练完成
    if local_rank == 0:
        logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()