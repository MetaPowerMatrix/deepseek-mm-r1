"""
Large Language Model (LLM) 训练脚本
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3ForCausalLM
from modeling_deepseek_副本 import DeepseekForCausalLM


# --------------------- 配置参数 ---------------------
class TrainingConfig:
	# 模型参数
	model_name = "deepseek-r2"  # 预训练模型名称
	max_seq_length = 512  # 输入序列最大长度

	# 训练参数
	batch_size = 8  # 单GPU批次大小
	num_epochs = 10  # 训练轮次
	learning_rate = 2e-5  # 初始学习率
	weight_decay = 0.01  # 权重衰减
	warmup_steps = 500  # 学习率预热步数
	gradient_accumulation = 4  # 梯度累积步数
	max_grad_norm = 1.0  # 梯度裁剪阈值

	# 设备设置
	fp16 = True  # 启用混合精度训练
	num_gpus = torch.cuda.device_count()  # 自动检测GPU数量

	# 路径设置
	dataset_path = "./data/train.txt"  # 训练数据路径
	output_dir = "./checkpoints"  # 模型保存路径
	log_dir = "./logs"  # 日志保存路径


# --------------------- 自定义数据集 ---------------------
class TextDataset(Dataset):
	def __init__(self, tokenizer, file_path, block_size):
		self.examples = []
		with open(file_path, "r", encoding="utf-8") as f:
			text = f.read()

		# 将文本分块为固定长度序列
		tokenized_text = tokenizer.encode(text)
		for i in range(0, len(tokenized_text) - block_size + 1, block_size):
			self.examples.append(tokenized_text[i:i + block_size])

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		return torch.tensor(self.examples[idx], dtype=torch.long)


# --------------------- 数据加载器 ---------------------
def create_dataloader(config, tokenizer):
	dataset = TextDataset(
		tokenizer=tokenizer,
		file_path=config.dataset_path,
		block_size=config.max_seq_length
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
	# 加载预训练模型和分词器
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)

	model = DeepseekV3ForCausalLM(DeepseekV3Config.from_json_file("config.json"))

	# 分布式训练设置
	if config.num_gpus > 1:
		model = nn.DataParallel(model)

	model.to(config.device)
	return model, tokenizer


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

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=config.warmup_steps,
		num_training_steps=total_steps
	)

	return optimizer, scheduler


# --------------------- 训练循环 ---------------------
def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, epoch):
	model.train()
	total_loss = 0.0
	accumulated_loss = 0.0

	for step, batch in enumerate(dataloader):
		inputs = batch.to(config.device)
		labels = inputs.clone()

		# 混合精度训练上下文
		with torch.cuda.amp.autocast(enabled=config.fp16):
			outputs = model(inputs, labels=labels)
			loss = outputs.loss
			loss = loss.mean()  # 多GPU平均

		# 梯度累积
		scaled_loss = loss / config.gradient_accumulation
		# 方向传播
		scaler.scale(scaled_loss).backward()

		# 梯度裁剪
		if (step + 1) % config.gradient_accumulation == 0:
			scaler.unscale_(optimizer)
			nn.utils.clip_grad_norm_(
				model.parameters(),
				config.max_grad_norm
			)

			# 参数更新
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
			scheduler.step()

		# 统计指标
		accumulated_loss += loss.item()
		total_loss += loss.item()

		# 每100步打印日志
		if step % 100 == 0:
			avg_loss = accumulated_loss / 100
			lr = optimizer.param_groups[0]["lr"]
			logging.info(
				f"Epoch {epoch} | Step {step} | "
				f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
			)
			accumulated_loss = 0.0

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
	writer = SummaryWriter(config.log_dir)  # TensorBoard

	# 初始化模型和分词器
	model, tokenizer = initialize_model(config)

	# 准备数据加载器
	train_loader = create_dataloader(config, tokenizer)
	total_steps = len(train_loader) * config.num_epochs

	# 初始化优化器和调度器
	optimizer, scheduler = create_optimizer_scheduler(model, config, total_steps)

	# 混合精度梯度缩放器
	scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

	# 训练循环
	for epoch in range(config.num_epochs):
		logging.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

		avg_loss = train_epoch(
			model, train_loader, optimizer,
			scheduler, scaler, config, epoch
		)

		# 记录指标
		writer.add_scalar("Loss/train", avg_loss, epoch)
		writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

		# 保存检查点
		if (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
			save_path = os.path.join(
				config.output_dir,
				f"checkpoint_epoch_{epoch}.pt"
			)
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': avg_loss,
			}, save_path)
			logging.info(f"Checkpoint saved to {save_path}")

	# 保存最终模型
	final_model_path = os.path.join(config.output_dir, "final_model")
	model.module.save_pretrained(final_model_path)  # 多GPU处理
	tokenizer.save_pretrained(final_model_path)
	logging.info(f"Model saved to {final_model_path}")


if __name__ == "__main__":
	main()