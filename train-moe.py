import os
import time
import logging
import math
import json
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

# 配置参数
class TrainingConfig:
    # 数据参数
    data_dir = "./data"
    train_file = "train.jsonl"
    val_file = "val.jsonl"
    max_seq_len = 512
    
    # 训练参数
    batch_size = 16  # 每个GPU的batch size
    gradient_accum_steps = 2  # 梯度累积步数
    epochs = 10
    learning_rate = 5e-5
    warmup_steps = 1000
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # 日志和保存
    output_dir = "./output"
    log_interval = 10  # 步数
    save_interval = 1000  # 步数
    eval_interval = 500  # 步数
    
    # 分布式
    world_size = torch.cuda.device_count()
    
    # 模型参数 (应与你的模型实现匹配)
    model_config = {
        "vocab_size": 50257,
        "pad_token_id": 50256
    }

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        file_ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext == '.jsonl':
                # 处理jsonl格式文件
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if 'text' in data:
                                self.examples.append(data['text'])
                        except json.JSONDecodeError:
                            continue
            else:
                # 处理普通文本文件
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(line)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# 分布式训练初始化
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# 学习率调度器
def get_scheduler(optimizer, warmup_steps, total_training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 训练步骤
def train_step(batch, model, optimizer, grad_accum_steps, global_step):
    input_ids = batch['input_ids'].cuda()
    attention_mask = batch['attention_mask'].cuda()
    labels = input_ids.clone()
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    loss = loss / grad_accum_steps  # 梯度累积
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    if (global_step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item() * grad_accum_steps  # 返回实际损失值

# 验证步骤
def evaluate(model, val_loader, local_rank):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            batch_size = input_ids.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size
    
    # 跨GPU聚合指标
    dist.all_reduce(torch.tensor(total_loss).cuda())
    dist.all_reduce(torch.tensor(total_samples).cuda())
    
    avg_loss = total_loss / total_samples
    return avg_loss

# 保存检查点
def save_checkpoint(model, optimizer, scheduler, global_step, epoch, output_dir, local_rank):
    if local_rank != 0:
        return
    
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch
    }
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        output_dir,
        f"checkpoint_step{global_step}.pt"
    )
    torch.save(checkpoint, checkpoint_path)

# 主训练函数
def train_model(model, tokenizer, local_rank):
    # 准备输出目录
    if local_rank == 0:
        os.makedirs(TrainingConfig.output_dir, exist_ok=True)
    
    # 初始化日志
    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(TrainingConfig.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training with config: {TrainingConfig.__dict__}")
    else:
        logger = None
    
    # 加载数据集
    train_dataset = TextDataset(
        os.path.join(TrainingConfig.data_dir, TrainingConfig.train_file),
        tokenizer,
        TrainingConfig.max_seq_len
    )
    val_dataset = TextDataset(
        os.path.join(TrainingConfig.data_dir, TrainingConfig.val_file),
        tokenizer,
        TrainingConfig.max_seq_len
    )
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 计算总训练步数
    total_steps_per_epoch = len(train_loader) // TrainingConfig.gradient_accum_steps
    total_training_steps = total_steps_per_epoch * TrainingConfig.epochs
    
    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.learning_rate,
        weight_decay=TrainingConfig.weight_decay
    )
    scheduler = get_scheduler(
        optimizer,
        TrainingConfig.warmup_steps,
        total_training_steps
    )
    
    # 训练状态
    global_step = 0
    start_time = time.time()
    
    # 训练循环
    for epoch in range(TrainingConfig.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch + 1}/{TrainingConfig.epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # 训练步骤
            loss = train_step(
                batch,
                model,
                optimizer,
                TrainingConfig.gradient_accum_steps,
                global_step
            )
            
            # 更新学习率
            if (global_step + 1) % TrainingConfig.gradient_accum_steps == 0:
                scheduler.step()
                global_step += 1
                
                # 日志记录
                if global_step % TrainingConfig.log_interval == 0 and local_rank == 0:
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step}/{total_training_steps} | "
                        f"Loss: {loss:.4f} | LR: {current_lr:.2e} | "
                        f"Elapsed: {elapsed:.2f}s"
                    )
                
                # 验证
                if global_step % TrainingConfig.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, local_rank)
                    if local_rank == 0:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                
                # 保存检查点
                if global_step % TrainingConfig.save_interval == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        TrainingConfig.output_dir,
                        local_rank
                    )
                    if local_rank == 0:
                        logger.info(f"Checkpoint saved at step {global_step}")
    
    # 训练结束
    if local_rank == 0:
        final_path = os.path.join(TrainingConfig.output_dir, "final_model.pt")
        torch.save(model.module.state_dict(), final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")

# 主函数
def main():
    # 初始化分布式训练
    local_rank = setup_distributed()
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化模型 (假设你的模型类名为TransformerMoE)
    from simple_moe import TransformerMoE  # 替换为你的模型导入
    
    model = TransformerMoE(
        vocab_size=TrainingConfig.model_config['vocab_size'],
        pad_token_id=TrainingConfig.model_config['pad_token_id']
    ).cuda()
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank])
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # 开始训练
    train_model(model, tokenizer, local_rank)

if __name__ == "__main__":
    main()