# train/train.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Adding to sys.path:", project_root)
sys.path.append(project_root)

import glob
import torch
from torch.utils.data import DataLoader
from data.dataset import TextDataset
from model.gpt import GPT
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

# 在外部加载模型和分词器
model = GPT2LMHeadModel.from_pretrained('gpt2')


def train():
	# 加载分词器
	tokenizer = AutoTokenizer.from_pretrained('gpt2')

	# 设置设备
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# 配置数据集
	dataset = TextDataset(directory_path="data/tokenized", seq_length=128, tokenizer=tokenizer)
	data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

	# 配置优化器
	optimizer = AdamW(model.parameters(), lr=1e-5)
	scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch减少学习率

	# 训练模型
	model.train()
	for epoch in range(3):  # 假设训练3个epoch
		total_loss = 0
		for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
			optimizer.zero_grad()

			input_ids = input_ids.to(device)
			target_ids = target_ids.to(device)

			# 前向传播
			outputs = model(input_ids, labels=target_ids)
			loss = outputs.loss
			total_loss += loss.item()

			# 反向传播
			loss.backward()
			# 训练过程中添加梯度裁剪
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

		# 输出每个 epoch 的损失
		print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

		# 学习率调度
		scheduler.step()

	# 保存训练好的模型
	torch.save(model.state_dict(), "trained_model/model.pth")

	tokenizer.save_pretrained("trained_model")


if __name__ == "__main__":
	train()
	