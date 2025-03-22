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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import argparse

# 导入MoE模型和数据集
from simple_moe import SimpleMoE, TransformerMoE, count_parameters
from dataset_moe import MoEDataset, create_dataloader, Tokenizer


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
    vocab_path = "./data/tokenizer.json"  # 词表路径（使用tokenizer.json作为词表）
    output_dir = "./checkpoints_moe"  # 模型保存路径
    log_dir = "./logs_moe"  # 日志保存路径
    
    # 数据设置
    text_file = None  # 原始文本文件，如果提供，则从该文件创建数据集
    jsonl_file = "./data/distill_r1_110k_sft.jsonl"  # 原始JSONL文件，如果提供，则从该文件创建数据集
    jsonl_sample_limit = 5000  # 限制处理的JSONL样本数量，None表示不限制


# --------------------- 数据加载器封装 ---------------------
def prepare_dataloader(config):
    """准备数据加载器"""
    # 加载或创建分词器
    tokenizer = None
    
    if config.model_type == "transformer_moe":
        # 首先尝试加载指定的tokenizer.json文件
        if os.path.exists(config.vocab_path):
            try:
                logging.info(f"从{config.vocab_path}加载词表...")
                tokenizer = Tokenizer.from_file(config.vocab_path)
                logging.info(f"成功加载词表，大小: {len(tokenizer.token_to_id)}")
            except Exception as e:
                logging.warning(f"从{config.vocab_path}加载词表失败: {str(e)}")
                
                # 尝试加载默认词表
                tokenizer = Tokenizer.create_default()
        else:
            logging.info(f"词表文件{config.vocab_path}不存在，尝试创建默认词表")
            # 确保目录存在
            os.makedirs(os.path.dirname(config.vocab_path), exist_ok=True)
            tokenizer = Tokenizer.create_default(save_path=config.vocab_path)
        
        # 更新配置中的词表大小
        config.vocab_size = len(tokenizer.token_to_id)
        logging.info(f"词表大小设置为: {config.vocab_size}")
    
    # 检查数据集文件是否存在
    dataset_exists = os.path.exists(config.dataset_path)
    
    # 如果提供了JSONL文件，且数据集不存在或JSONL文件比数据集更新，则从JSONL创建数据集
    if config.jsonl_file and os.path.exists(config.jsonl_file):
        create_from_jsonl = False
        
        if not dataset_exists:
            logging.info(f"数据集不存在，将从JSONL文件创建: {config.jsonl_file}")
            create_from_jsonl = True
        elif os.path.getmtime(config.jsonl_file) > os.path.getmtime(config.dataset_path):
            logging.info(f"JSONL文件比数据集更新，将重新创建数据集: {config.jsonl_file}")
            create_from_jsonl = True
        
        if create_from_jsonl:
            if config.model_type == "transformer_moe":
                logging.info(f"从JSONL文件创建TransformerMoE数据集: {config.jsonl_file}")
                MoEDataset.create_from_jsonl(
                    jsonl_path=config.jsonl_file,
                    output_path=config.dataset_path,
                    tokenizer=tokenizer,
                    max_seq_length=config.max_seq_length,
                    limit=config.jsonl_sample_limit
                )
            else:
                logging.warning("仅TransformerMoE模型支持从JSONL创建数据集")
    # 如果提供了原始文本文件，且数据集不存在或文本文件比数据集更新，则从文本创建数据集
    elif config.text_file and os.path.exists(config.text_file):
        create_from_text = False
        
        if not dataset_exists:
            logging.info(f"数据集不存在，将从文本文件创建: {config.text_file}")
            create_from_text = True
        elif os.path.getmtime(config.text_file) > os.path.getmtime(config.dataset_path):
            logging.info(f"文本文件比数据集更新，将重新创建数据集: {config.text_file}")
            create_from_text = True
        
        if create_from_text:
            if config.model_type == "transformer_moe":
                logging.info(f"从文本文件创建TransformerMoE数据集: {config.text_file}")
                MoEDataset.create_from_text(
                    file_path=config.text_file,
                    output_path=config.dataset_path,
                    tokenizer=tokenizer,
                    block_size=config.max_seq_length
                )
            else:
                logging.warning("仅TransformerMoE模型支持从文本创建数据集")
    
    # 创建数据集
    dataset = MoEDataset(
        file_path=config.dataset_path,
        max_seq_length=config.max_seq_length,
        model_type=config.model_type,
        tokenizer=tokenizer
    )
    
    # 创建数据加载器
    return create_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_gpus=config.num_gpus,
        shuffle=True,
        num_workers=4
    ), tokenizer


# --------------------- 模型初始化 ---------------------
def initialize_model(config, tokenizer=None):
    if config.model_type == "simple_moe":
        model = SimpleMoE(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_experts=config.num_experts,
            k=config.k
        )
    else:  # transformer_moe
        # 如果有分词器，使用分词器的词表大小
        if tokenizer:
            vocab_size = len(tokenizer.token_to_id)
            # 更新配置
            config.vocab_size = vocab_size
            logging.info(f"使用分词器词表大小: {vocab_size}")
        else:
            vocab_size = config.vocab_size
            
        model = TransformerMoE(
            vocab_size=vocab_size,
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
            avg_loss = accumulated_loss / (step % 50 + 1) if step % 50 > 0 else accumulated_loss
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


# --------------------- 文本生成函数 (用于TransformerMoE) ---------------------
def generate_text(model, tokenizer, start_text, max_length=100, temperature=1.0, device="cpu"):
    """
    使用训练好的TransformerMoE模型生成文本
    
    参数:
        model: 训练好的TransformerMoE模型
        tokenizer: 分词器
        start_text: 生成的起始文本
        max_length: 最大生成长度
        temperature: 采样温度，越高越随机
        device: 计算设备
    
    返回:
        生成的文本
    """
    if not tokenizer:
        logging.error("生成文本需要分词器")
        return start_text
    
    model.eval()
    
    # 将起始文本转换为token ids
    input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型预测
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接到输入序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # 如果生成了结束符，提前结束
            if next_token.item() == tokenizer.special_tokens.get("<eos>", -1):
                break
    
    # 解码生成的token序列
    output_text = tokenizer.decode(input_ids[0].tolist())
    
    return output_text


# --------------------- 主函数 ---------------------
def main():
    # 初始化配置
    config = TrainingConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.vocab_path), exist_ok=True)
    
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
    
    # 准备数据加载器和分词器
    train_loader, tokenizer = prepare_dataloader(config)
    
    # 初始化模型
    model = initialize_model(config, tokenizer)
    
    # 获取损失函数
    loss_fn = get_loss_fn(config)
    
    # 计算总步数
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
        
        # 对于TransformerMoE，生成一些示例文本
        if config.model_type == "transformer_moe" and tokenizer:
            if (epoch + 1) % 2 == 0:  # 每2个epoch生成一次
                start_text = "今天是"
                generated_text = generate_text(
                    model, tokenizer, start_text, 
                    max_length=50, device=config.device
                )
                logging.info(f"生成的文本示例: {generated_text}")
        
        # 保存检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                config.output_dir,
                f"best_model_{config.model_type}.pt"
            )
            
            # 保存模型和配置
            model_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }
            
            # 如果有分词器，也保存它
            if tokenizer:
                tokenizer.save_vocab(config.vocab_path)
                model_data['vocab_path'] = config.vocab_path
            
            torch.save(model_data, save_path)
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
        'config': vars(config),
        'vocab_path': config.vocab_path if tokenizer else None
    }, final_model_path)
    
    logging.info(f"训练完成! 最终模型保存到 {final_model_path}")
    logging.info(f"最佳损失: {best_loss:.4f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练MoE模型")
    parser.add_argument("--process-jsonl", action="store_true", help="仅处理JSONL文件，不训练模型")
    parser.add_argument("--limit", type=int, default=5000, help="处理JSONL文件的样本数限制")
    parser.add_argument("--train", action="store_true", help="训练模型")
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    if args.process_jsonl:
        # 仅处理JSONL文件
        config = TrainingConfig()
        config.jsonl_sample_limit = args.limit
        
        # 创建分词器
        if os.path.exists(config.vocab_path):
            try:
                tokenizer = Tokenizer.from_file(config.vocab_path)
                logging.info(f"从{config.vocab_path}加载词表，大小: {len(tokenizer.token_to_id)}")
            except Exception as e:
                logging.warning(f"加载词表失败: {str(e)}")
                tokenizer = Tokenizer.create_default(save_path=config.vocab_path)
        else:
            tokenizer = Tokenizer.create_default(save_path=config.vocab_path)
        
        # 处理JSONL文件
        logging.info(f"开始处理JSONL文件: {config.jsonl_file}")
        logging.info(f"样本限制: {config.jsonl_sample_limit}")
        
        count = MoEDataset.create_from_jsonl(
            jsonl_path=config.jsonl_file,
            output_path=config.dataset_path,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            limit=config.jsonl_sample_limit
        )
        
        logging.info(f"JSONL处理完成，共处理{count}个样本")
    elif args.train:
        # 训练模型
        main()
    else:
        # 默认行为，运行main函数
        main() 