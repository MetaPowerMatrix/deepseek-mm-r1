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
from tqdm import tqdm

# 导入PyTorch分布式训练相关库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入MoE模型和数据集
from simple_moe import SimpleMoE, TransformerMoE, count_parameters
from dataset_moe import MoEDataset, create_dataloader, Tokenizer


# --------------------- 分布式训练初始化 ---------------------
def setup_distributed(rank, world_size, port=29500):
    """
    初始化分布式训练环境
    
    参数:
        rank: 当前进程的rank
        world_size: 总进程数（GPU数量）
        port: 通信端口
    """
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = str(port)
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    """检查是否处于分布式训练模式"""
    return dist.is_initialized()


def get_rank():
    """获取当前进程的rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """检查是否为主进程（rank 0）"""
    return get_rank() == 0


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
def prepare_dataloader(model_type, data_path, batch_size, vocab_path=None, vocab_size=30000, max_seq_length=512, limit=None, num_workers=4, distributed=False, rank=0, world_size=1):
    """
    准备数据加载器
    
    参数:
        model_type: 模型类型，"simple_moe"或"transformer_moe"
        data_path: 数据集路径
        batch_size: 批次大小
        vocab_path: 词表路径（仅TransformerMoE需要）
        vocab_size: 词表大小
        max_seq_length: 最大序列长度（仅TransformerMoE需要）
        limit: 限制处理的样本数量
        num_workers: 数据加载线程数
        distributed: 是否使用分布式训练
        rank: 当前进程的rank
        world_size: 总进程数
    
    返回:
        dataloader, vocab_size（用于更新模型配置）
    """
    # 加载或创建词表
    tokenizer = None
    if model_type != "simple_moe":
        if vocab_path and os.path.exists(vocab_path):
            # 仅在主进程中打印信息
            if not distributed or rank == 0:
                logging.info(f"从{vocab_path}加载词表")
            from dataset_moe import Tokenizer
            tokenizer = Tokenizer.from_file(vocab_path)
            # 更新词表大小
            vocab_size = max(vocab_size, len(tokenizer.token_to_id))
            if not distributed or rank == 0:
                logging.info(f"词表大小: {vocab_size}")
        else:
            if not distributed or rank == 0:
                logging.warning(f"未找到词表文件{vocab_path}，将使用默认词表")
    
    # 确保数据目录存在
    if not distributed or rank == 0:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # 如果是分布式训练，需要等待所有进程同步
    if distributed:
        dist.barrier()
    
    # 如果是JSONL文件，创建数据集
    if data_path.endswith(".jsonl"):
        from dataset_moe import MoEDataset
        dataset_path = data_path.replace(".jsonl", ".pt")
        if not os.path.exists(dataset_path):
            if not distributed or rank == 0:
                logging.info(f"从{data_path}创建数据集")
            if model_type == "simple_moe":
                # SimpleMoE没有实现JSONL加载，生成合成数据集
                if not distributed or rank == 0:
                    logging.warning(f"SimpleMoE不支持JSONL格式，创建合成数据集")
                dataset = MoEDataset(
                    file_path="./data/simple_moe_synthetic.pt",
                    model_type=model_type
                )
            else:
                # 从JSONL创建数据集（仅在主进程中处理）
                if not distributed or rank == 0:
                    MoEDataset.create_from_jsonl(
                        jsonl_path=data_path,
                        output_path=dataset_path,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        limit=limit
                    )
                
                # 如果是分布式训练，确保所有进程等待主进程创建完数据集
                if distributed:
                    dist.barrier()
                
                dataset = MoEDataset(
                    file_path=dataset_path,
                    max_seq_length=max_seq_length,
                    model_type=model_type,
                    tokenizer=tokenizer
                )
        else:
            # 数据集已存在，直接加载
            if not distributed or rank == 0:
                logging.info(f"从{dataset_path}加载数据集")
            dataset = MoEDataset(
                file_path=dataset_path,
                max_seq_length=max_seq_length,
                model_type=model_type,
                tokenizer=tokenizer
            )
    else:
        # 直接加载.pt数据集
        from dataset_moe import MoEDataset
        dataset = MoEDataset(
            file_path=data_path,
            max_seq_length=max_seq_length if model_type != "simple_moe" else None,
            model_type=model_type,
            tokenizer=tokenizer if model_type != "simple_moe" else None
        )
    
    # 创建数据加载器
    # 对于分布式训练，使用DistributedSampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的批次
    )
    
    if not distributed or rank == 0:
        logging.info(f"数据集大小: {len(dataset)}，批次数: {len(dataloader)}")
    
    # 返回dataloader和更新后的vocab_size
    return dataloader, vocab_size, sampler


# --------------------- 模型初始化 ---------------------
def initialize_model(model_type, config, distributed=False, local_rank=0):
    """
    初始化模型
    
    参数:
        model_type: 模型类型，"simple_moe"或"transformer_moe"
        config: 配置对象
        distributed: 是否为分布式训练
        local_rank: 本地GPU的rank
    
    返回:
        初始化的模型（如果分布式训练则返回DDP封装的模型）
    """
    from simple_moe import SimpleMoE, TransformerMoE, LongContextTransformerMoE
    
    if not distributed or local_rank == 0:
        logging.info(f"初始化{model_type}模型...")
    model = None
    
    if model_type == "simple_moe":
        model = SimpleMoE(
            input_size=config.input_size,
            hidden_size=config.hidden_size, 
            output_size=config.output_size,
            num_experts=config.num_experts,
            k=config.k
        )
    elif model_type == "transformer_moe":
        # 确保vocab_size已正确设置
        # 确保vocab_size至少为1000（防止词表过小导致问题）
        vocab_size = max(1000, config.vocab_size)
        model = TransformerMoE(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_length,
            num_experts=config.num_experts,
            k=config.k,
            dropout=config.dropout
        )
    elif model_type == "long_transformer_moe":
        # 确保vocab_size已正确设置
        vocab_size = max(1000, config.vocab_size)
        model = LongContextTransformerMoE(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_length,
            num_experts=config.num_experts,
            k=config.k,
            dropout=config.dropout,
            scaling_factor=config.scaling_factor,
            rope_theta=10000
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    if model is not None:
        # 设置正确的设备
        device = torch.device("cuda", local_rank) if distributed else getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        
        # 如果是分布式训练，使用DDP包装模型
        if distributed:
            # 为了防止由于默认流同步导致死锁，在DDP包装之前设置find_unused_parameters=False
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
        # 打印模型参数数量
        if not distributed or local_rank == 0:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            logging.info(f"模型参数数量: {num_params:.2f}M")
    
    return model


# --------------------- 损失函数 ---------------------
def get_loss_fn(config):
    if config.model_type == "simple_moe":
        # 回归任务使用MSE损失
        return nn.MSELoss().to(config.device)
    else:
        # 语言模型使用交叉熵损失
        return nn.CrossEntropyLoss().to(config.device)


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
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, 
              gradient_accumulation_steps=1, mixed_precision=False, amp_scaler=None,
              clip_grad=1.0, model_type="transformer_moe", distributed=False, local_rank=0, sampler=None):
    """单个训练轮次"""
    model.train()
    total_loss = 0.0
    batches_processed = 0
    
    # 如果是分布式训练，需要设置sampler的epoch
    if distributed and sampler is not None:
        sampler.set_epoch(epoch)
    
    # 创建tqdm进度条 (仅在主进程中显示)
    if not distributed or local_rank == 0:
        progress_bar = tqdm(total=len(dataloader), desc=f"训练中")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 将数据移动到指定设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 确保输入索引在有效范围内
        if model_type in ["transformer_moe", "long_transformer_moe"]:
            # 注意：如果模型是DDP封装的，需要访问.module
            vocab_size = model.module.token_embedding.weight.size(0) if distributed else model.token_embedding.weight.size(0)
            # 检查并裁剪超出范围的索引
            inputs = torch.clamp(inputs, 0, vocab_size - 1)
            targets = torch.clamp(targets, 0, vocab_size - 1)
        
        # 在梯度累积步骤开始时清零梯度
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 混合精度训练
        if mixed_precision and amp_scaler is not None:
            with torch.cuda.amp.autocast():
                # 向前传播
                outputs = model(inputs)
                
                # 计算损失
                if model_type == "simple_moe":
                    loss = criterion(outputs, targets)
                else:  # transformer_moe
                    # 对于TransformerMoE，输出形状为[batch_size, seq_len, vocab_size]
                    # 调整为计算CrossEntropyLoss所需的形状
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                
                # 缩放损失以适应梯度累积
                loss = loss / gradient_accumulation_steps
            
            # 反向传播
            amp_scaler.scale(loss).backward()
                
            # 如果达到累积步数，则更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # 对梯度进行裁剪，防止梯度爆炸
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    
                # 使用scaler更新权重
                amp_scaler.step(optimizer)
                amp_scaler.update()
                
                # 更新学习率
                scheduler.step()
        else:
            # 非混合精度训练
            # 向前传播
            outputs = model(inputs)
            
            # 计算损失
            if model_type == "simple_moe":
                loss = criterion(outputs, targets)
            else:  # transformer_moe
                # 对于TransformerMoE，输出形状为[batch_size, seq_len, vocab_size]
                # 调整为计算CrossEntropyLoss所需的形状
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
            
            # 缩放损失以适应梯度累积
            loss = loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 如果达到累积步数，则更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # 对梯度进行裁剪，防止梯度爆炸
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                # 更新权重
                optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                # 清零梯度
                optimizer.zero_grad()
        
        # 累计损失
        total_loss += loss.item() * gradient_accumulation_steps
        batches_processed += 1
        
        # 更新进度条 (仅在主进程中)
        if not distributed or local_rank == 0:
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({"loss": f"{total_loss / batches_processed:.4f}"})
            progress_bar.update(1)
    
    # 关闭进度条 (仅在主进程中)
    if not distributed or local_rank == 0:
        progress_bar.close()
    
    # 计算平均损失
    avg_loss = total_loss / batches_processed
    
    # 如果是分布式训练，需要在所有进程间同步损失
    if distributed:
        # 收集所有GPU上的损失值
        loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    
    return avg_loss


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


# --------------------- 辅助函数 ---------------------
def setup_logging(log_file=None):
    """设置日志"""
    log_level = logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # 设置根日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # 如果提供了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练MoE模型")
    
    # 模型与训练设置
    parser.add_argument("--model_type", type=str, default="transformer_moe", 
                        choices=["simple_moe", "transformer_moe", "long_transformer_moe"],
                        help="模型类型: simple_moe, transformer_moe, long_transformer_moe")
    parser.add_argument("--data_path", type=str, default="./data/distill_r1_110k_sft.jsonl",
                        help="训练数据路径（.pt文件或.jsonl文件）")
    parser.add_argument("--vocab_path", type=str, default="./data/tokenizer.json",
                        help="词表路径")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_moe",
                        help="模型保存目录")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="训练轮次")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="初始学习率")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true",
                        help="是否启用混合精度训练")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout比率")
                        
    # 模型结构参数
    parser.add_argument("--d_model", type=int, default=768,
                        help="模型隐藏层维度")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Transformer层数")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="前馈网络维度")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="专家数量")
    parser.add_argument("--k", type=int, default=2,
                        help="每次激活的专家数量")
                        
    # 长上下文相关参数
    parser.add_argument("--scaling_factor", type=float, default=8.0,
                        help="RoPE缩放因子，用于长上下文模型")
    
    # 检查点与评估
    parser.add_argument("--eval_every", type=int, default=500,
                        help="每多少步评估一次")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="用于恢复训练的检查点路径")
    parser.add_argument("--train", action="store_true",
                        help="训练模型")
    parser.add_argument("--process-jsonl", action="store_true", help="仅处理JSONL文件，不训练模型")
    parser.add_argument("--limit", type=int, default=5000, help="处理JSONL文件的样本数限制")
    parser.add_argument("--distributed", action="store_true", help="是否启用分布式训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU的rank")
    parser.add_argument("--world_size", type=int, help="总进程数（GPU数量）")
    parser.add_argument("--master_port", type=int, default=29500, help="通信端口")
    
    return parser.parse_args()


# --------------------- 主函数 ---------------------
def main():
    # 声明全局变量
    global writer, tokenizer
    
    # 解析命令行参数
    args = parse_args()
    
    # 处理分布式训练参数
    distributed = args.distributed or args.local_rank != -1
    local_rank = args.local_rank
    world_size = args.world_size or torch.cuda.device_count()
    
    # 初始化分布式环境
    if distributed:
        if local_rank == -1:
            # 如果未指定local_rank但启用了分布式训练，退出
            logging.error("分布式训练需要指定local_rank")
            exit(1)
        
        # 初始化分布式环境
        setup_distributed(local_rank, world_size, args.master_port)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置配置
    config = TrainingConfig()
    config.model_type = args.model_type
    config.device = device
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.save_dir = args.save_dir
    config.eval_every = args.eval_every
    config.vocab_path = args.vocab_path
    config.checkpoint_path = args.checkpoint_path
    
    # 设置dropout
    config.dropout = args.dropout
    
    # 从参数更新模型配置
    if args.d_model:
        config.d_model = args.d_model
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.d_ff:
        config.d_ff = args.d_ff
    if args.num_experts:
        config.num_experts = args.num_experts
    if args.k:
        config.k = args.k
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.gradient_accumulation:
        config.gradient_accumulation = args.gradient_accumulation
    if args.fp16 is not None:
        config.fp16 = args.fp16
    
    # 创建保存目录 (仅在主进程中)
    if not distributed or local_rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置日志 (仅在主进程中)
    if not distributed or local_rank == 0:
        setup_logging(os.path.join(config.save_dir, "training.log"))
    
    # 设置随机种子（确保所有进程使用相同种子）
    setup_seed(42 + local_rank if distributed else 42)
    
    # 设置TensorBoard（仅在主进程中）
    if not distributed or local_rank == 0:
        log_dir = os.path.join(config.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard日志保存在: {log_dir}")
    
    # 准备数据加载器并获取更新后的vocab_size
    train_loader, updated_vocab_size, sampler = prepare_dataloader(
        model_type=config.model_type,
        data_path=config.data_path,
        batch_size=config.batch_size,
        vocab_path=config.vocab_path,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        distributed=distributed,
        rank=local_rank,
        world_size=world_size
    )
    
    # 如果是文本模型，获取tokenizer用于生成示例 (仅在主进程中)
    tokenizer = None
    if config.model_type != "simple_moe" and os.path.exists(config.vocab_path):
        try:
            from dataset_moe import Tokenizer
            tokenizer = Tokenizer.from_file(config.vocab_path)
            if not distributed or local_rank == 0:
                logging.info(f"为文本生成加载词表，大小: {len(tokenizer.token_to_id)}")
        except Exception as e:
            if not distributed or local_rank == 0:
                logging.warning(f"加载生成用词表失败: {str(e)}")
    
    # 更新config中的vocab_size
    if config.model_type != "simple_moe":
        if updated_vocab_size != config.vocab_size:
            if not distributed or local_rank == 0:
                logging.info(f"更新词表大小: {config.vocab_size} -> {updated_vocab_size}")
            config.vocab_size = updated_vocab_size
    
    # 初始化模型
    model = initialize_model(config.model_type, config, distributed, local_rank)
    
    # 获取损失函数
    loss_fn = get_loss_fn(config)
    
    # 计算总步数
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    
    # 初始化优化器和调度器
    optimizer, scheduler = create_optimizer_scheduler(model, config, total_steps)
    
    # 混合精度梯度缩放器
    # 仅在CUDA可用时启用混合精度
    if torch.cuda.is_available():
        config.fp16 = config.fp16  # 保持现有设置
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        if not distributed or local_rank == 0:
            logging.info(f"混合精度训练: {'启用' if config.fp16 else '禁用'}")
    else:
        config.fp16 = False  # 在CPU上禁用混合精度
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        if not distributed or local_rank == 0:
            logging.info("在CPU上运行，已禁用混合精度训练")
    
    # 训练循环
    if not distributed or local_rank == 0:
        logging.info(f"开始训练 {config.model_type} 模型...")
        logging.info(f"数据集大小: {len(train_loader.dataset)} 样本")
        effective_batch = config.batch_size * (world_size if distributed else 1)
        logging.info(f"批次大小: {config.batch_size} (每GPU) x {world_size if distributed else 1} GPU = {effective_batch} (有效)")
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        if not distributed or local_rank == 0:
            logging.info(f"开始 Epoch {epoch + 1}/{config.num_epochs}")
        
        # 训练一个epoch
        avg_loss = train_epoch(
            model, train_loader, loss_fn, optimizer,
            scheduler, config.device, config.gradient_accumulation, 
            config.fp16, scaler, config.max_grad_norm, config.model_type,
            distributed, local_rank, sampler
        )
        
        # 记录指标 (仅在主进程中)
        if not distributed or local_rank == 0:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            logging.info(f"Epoch {epoch + 1} 完成: 平均损失 = {avg_loss:.4f}")
        
        # 对于TransformerMoE，生成一些示例文本 (仅在主进程中)
        if (not distributed or local_rank == 0) and config.model_type == "transformer_moe" and tokenizer:
            if (epoch + 1) % 2 == 0:  # 每2个epoch生成一次
                start_text = "今天是"
                # 在生成文本时，如果模型是DDP封装的，需要使用.module
                model_for_generation = model.module if distributed else model
                generated_text = generate_text(
                    model_for_generation, tokenizer, start_text, 
                    max_length=50, device=config.device
                )
                logging.info(f"生成的文本示例: {generated_text}")
        
        # 保存检查点 (仅在主进程中)
        if (not distributed or local_rank == 0) and avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                config.output_dir,
                f"best_model_{config.model_type}.pt"
            )
            
            # 获取模型状态，如果是DDP模型，获取内部模型
            model_to_save = model.module if distributed else model
            
            # 保存模型和配置
            model_data = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
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
        
        # 定期保存检查点 (仅在主进程中)
        if (not distributed or local_rank == 0) and (epoch + 1) % 2 == 0:
            save_path = os.path.join(
                config.output_dir,
                f"{config.model_type}_checkpoint_epoch_{epoch + 1}.pt"
            )
            
            # 获取模型状态，如果是DDP模型，获取内部模型
            model_to_save = model.module if distributed else model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }, save_path)
            logging.info(f"保存检查点到 {save_path}")
    
    # 保存最终模型 (仅在主进程中)
    if not distributed or local_rank == 0:
        final_model_path = os.path.join(
            config.output_dir,
            f"final_{config.model_type}_model.pt"
        )
        
        # 获取模型状态，如果是DDP模型，获取内部模型
        model_to_save = model.module if distributed else model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': vars(config),
            'vocab_path': config.vocab_path if tokenizer else None
        }, final_model_path)
        
        logging.info(f"训练完成! 最终模型保存到 {final_model_path}")
        logging.info(f"最佳损失: {best_loss:.4f}")
        
        writer.close()
    
    # 清理分布式环境
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志格式（在分布式训练中，只有主进程输出完整日志）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 检测是否启用分布式训练
    distributed = args.distributed or args.local_rank != -1
    
    # 如果是分布式模式，检查local_rank是否有效
    if distributed and args.local_rank == -1:
        print("分布式训练需要指定local_rank参数，请使用 --local_rank=<rank> 或使用torch.distributed.launch启动")
        exit(1)
    
    # 同时输出到控制台（仅主进程输出完整信息）
    if not distributed or args.local_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
    
    if args.process_jsonl:
        # 仅处理JSONL文件（只在主进程中执行）
        if not distributed or args.local_rank == 0:
            config = TrainingConfig()
            config.jsonl_sample_limit = args.limit if hasattr(args, 'limit') else 5000
            
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
        
        # 如果是分布式训练，等待主进程处理完成
        if distributed:
            dist.barrier()
    else:
        # 训练模型
        main()
        
        # 清理分布式环境
        if distributed:
            cleanup_distributed() 