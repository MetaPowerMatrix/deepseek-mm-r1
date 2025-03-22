"""
混合专家模型(Mixture of Experts)训练脚本
功能特性：
- 支持多GPU分布式训练
- 混合精度训练（FP16/FP32）
- 梯度累积
- 动态学习率调度
- 模型检查点保存
- 训练过程指标可视化
"""

import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# 导入MoE模型
from simple_moe import SimpleMoE, TransformerMoE, count_parameters


# --------------------- 配置参数 ---------------------
class TrainingConfig:
    # 模型参数
    model_type = "transformer_moe"  # 选择 "simple_moe" 或 "transformer_moe"
    
    # SimpleMoE参数
    input_size = 1024
    hidden_size = 2048
    output_size = 1024
    
    # TransformerMoE参数
    vocab_size = 30000
    d_model = 1024
    num_heads = 16
    num_layers = 6
    d_ff = 4096
    max_seq_length = 512
    
    # 通用MoE参数
    num_experts = 16
    k = 2  # 激活的专家数量
    
    # 训练参数
    batch_size = 8  # 单GPU批次大小
    num_epochs = 10  # 训练轮次
    learning_rate = 1e-4  # 初始学习率
    weight_decay = 0.01  # 权重衰减
    warmup_steps = 500  # 学习率预热步数
    gradient_accumulation = 4  # 梯度累积步数
    max_grad_norm = 1.0  # 梯度裁剪阈值

    # 设备设置
    fp16 = True  # 启用混合精度训练
    num_gpus = torch.cuda.device_count()  # 自动检测GPU数量

    # 路径设置
    dataset_path = "./data/train_data.pt"  # 训练数据路径
    output_dir = "./checkpoints_moe"  # 模型保存路径
    log_dir = "./logs_moe"  # 日志保存路径


# --------------------- 自定义数据集 ---------------------
class MoEDataset(Dataset):
    """用于MoE模型的数据集类"""
    
    def __init__(self, file_path, max_seq_length=None, model_type="simple_moe"):
        self.model_type = model_type
        
        if not os.path.exists(file_path):
            # 如果没有真实数据集，创建合成数据集
            self.create_synthetic_dataset(file_path, max_seq_length, model_type)
        
        # 加载数据集
        self.data = torch.load(file_path)
        
    def create_synthetic_dataset(self, file_path, max_seq_length, model_type):
        """创建合成数据集用于训练"""
        num_samples = 1000  # 合成样本数量
        
        if model_type == "simple_moe":
            # 为SimpleMoE创建向量数据
            data = torch.randn(num_samples, 1024)  # 随机输入特征
            labels = torch.randn(num_samples, 1024)  # 随机标签
            
            self.data = [(x, y) for x, y in zip(data, labels)]
            
        else:  # transformer_moe
            # 为TransformerMoE创建序列数据
            data = torch.randint(0, 30000, (num_samples, max_seq_length))  # 随机token序列
            labels = data.clone()  # 自回归任务中，输入即标签
            
            self.data = [(x, y) for x, y in zip(data, labels)]
        
        # 创建输出目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存数据集
        torch.save(self.data, file_path)
        logging.info(f"创建合成数据集: {file_path}, 样本数: {num_samples}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# --------------------- 数据加载器 ---------------------
def create_dataloader(config):
    dataset = MoEDataset(
        file_path=config.dataset_path,
        max_seq_length=config.max_seq_length,
        model_type=config.model_type
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=config.num_gpus,
        shuffle=True
    ) if config.num_gpus > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )
    return dataloader


# --------------------- 模型初始化 ---------------------
def initialize_model(config):
    if config.model_type == "simple_moe":
        model = SimpleMoE(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_experts=config.num_experts,
            k=config.k
        )
    else:  # transformer_moe
        model = TransformerMoE(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_length,
            num_experts=config.num_experts,
            k=config.k
        )
    
    # 输出模型参数统计
    logging.info(f"模型类型: {config.model_type}")
    logging.info(f"模型参数量: {count_parameters(model):.2f}M")
    
    # 分布式训练设置
    if config.num_gpus > 1:
        model = nn.DataParallel(model)
    
    model.to(config.device)
    return model


# --------------------- 损失函数 ---------------------
def get_loss_fn(config):
    if config.model_type == "simple_moe":
        # 回归任务使用MSE损失
        return nn.MSELoss()
    else:
        # 语言模型使用交叉熵损失
        return nn.CrossEntropyLoss()


# --------------------- 优化器与调度器 ---------------------
def create_optimizer_scheduler(model, config, total_steps):
    # 设置参数分组（不同层不同学习率）
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=1e-8
    )
    
    # 使用余弦学习率调度
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.learning_rate * 0.1
    )
    
    return optimizer, scheduler


# --------------------- 训练循环 ---------------------
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, config, epoch):
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        # 解包数据
        inputs, targets = batch
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        
        # 混合精度训练上下文
        with torch.cuda.amp.autocast(enabled=config.fp16):
            if config.model_type == "simple_moe":
                # SimpleMoE直接前向传播
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            else:
                # TransformerMoE处理
                outputs = model(inputs)
                # 重塑输出以适应交叉熵损失 [batch_size*seq_len, vocab_size]
                outputs = outputs.view(-1, config.vocab_size)
                targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
        
        # 梯度累积
        scaled_loss = loss / config.gradient_accumulation
        
        # 反向传播
        scaler.scale(scaled_loss).backward()
        
        # 梯度裁剪和更新
        if (step + 1) % config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # 记录损失
        accumulated_loss += loss.item()
        total_loss += loss.item()
        
        # 每50步打印一次日志
        if step % 50 == 0:
            items_seen = (step + 1) * config.batch_size
            avg_loss = accumulated_loss / (step % 50 + 1)
            lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch+1} | Step {step}/{len(dataloader)} | "
                f"Items: {items_seen} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
            )
            accumulated_loss = 0.0
    
    return total_loss / len(dataloader)


# --------------------- 评估函数 ---------------------
def evaluate(model, dataloader, loss_fn, config):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            if config.model_type == "simple_moe":
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            else:
                outputs = model(inputs)
                outputs = outputs.view(-1, config.vocab_size)
                targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# --------------------- 主函数 ---------------------
def main():
    # 初始化配置
    config = TrainingConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=os.path.join(config.log_dir, f"train_{timestamp}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    writer = SummaryWriter(os.path.join(config.log_dir, timestamp))
    
    # 记录训练配置
    logging.info("训练配置:")
    for key, value in vars(config).items():
        logging.info(f"  {key}: {value}")
    
    # 初始化模型
    model = initialize_model(config)
    
    # 获取损失函数
    loss_fn = get_loss_fn(config)
    
    # 准备数据加载器
    train_loader = create_dataloader(config)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    
    # 初始化优化器和调度器
    optimizer, scheduler = create_optimizer_scheduler(model, config, total_steps)
    
    # 混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    
    # 训练循环
    logging.info(f"开始训练 {config.model_type} 模型...")
    logging.info(f"数据集大小: {len(train_loader.dataset)} 样本")
    logging.info(f"批次大小: {config.batch_size} (单GPU) x {config.num_gpus} GPU = {config.batch_size * max(1, config.num_gpus)} (有效)")
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        logging.info(f"开始 Epoch {epoch + 1}/{config.num_epochs}")
        
        # 训练一个epoch
        avg_loss = train_epoch(
            model, train_loader, optimizer,
            scheduler, loss_fn, scaler, config, epoch
        )
        
        # 记录指标
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        logging.info(f"Epoch {epoch + 1} 完成: 平均损失 = {avg_loss:.4f}")
        
        # 保存检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                config.output_dir,
                f"best_model_{config.model_type}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }, save_path)
            logging.info(f"保存最佳模型到 {save_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(
                config.output_dir,
                f"{config.model_type}_checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }, save_path)
            logging.info(f"保存检查点到 {save_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(
        config.output_dir,
        f"final_{config.model_type}_model.pt"
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }, final_model_path)
    
    logging.info(f"训练完成! 最终模型保存到 {final_model_path}")
    logging.info(f"最佳损失: {best_loss:.4f}")
    
    writer.close()


if __name__ == "__main__":
    main() 