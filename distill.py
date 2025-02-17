"""
长链式思维蒸馏训练脚本
功能特性：
- 教师模型生成多步骤推理链
- 学生模型联合学习答案和推理步骤
- 输出长度控制正则化
- 混合精度训练
- 多GPU分布式支持
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


# --------------------- 配置参数 ---------------------
class TrainingConfig:
	# 模型参数
	teacher_model = "DeepSeek-R1"  # 教师模型名称/路径
	student_model = "DeepSeek-V3"  # 学生模型名称/路径
	max_teacher_length = 512  # 教师模型生成最大长度
	max_student_length = 256  # 学生模型生成最大长度

	# 训练参数
	batch_size = 8
	num_epochs = 10
	learning_rate = 3e-5
	answer_loss_weight = 0.7  # 最终答案损失权重
	step_loss_weight = 0.3  # 推理步骤损失权重
	length_penalty = 0.1  # 输出长度正则化系数

	# 设备设置
	fp16 = True
	num_gpus = torch.cuda.device_count()


# --------------------- 自定义数据集 ---------------------
class CoTDataset(Dataset):
	def __init__(self, questions, tokenizer, teacher_model, config):
		"""
		动态生成含长链式思维的数据
		"""
		self.tokenizer = tokenizer
		self.teacher_model = teacher_model
		self.config = config
		self.questions = questions

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):
		question = self.questions[idx]

		# 使用教师模型生成带推理链的答案
		with torch.no_grad():
			inputs = self.tokenizer(
				question,
				return_tensors="pt",
				max_length=self.config.max_teacher_length,
				truncation=True
			).to(self.teacher_model.device)

			outputs = self.teacher_model.generate(
				**inputs,
				max_length=self.config.max_teacher_length,
				do_sample=True,
				top_p=0.9,
				num_return_sequences=1
			)

		full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

		# 解析输出为问题和推理链（示例解析逻辑）
		if "步骤" in full_output:
			steps, answer = self._parse_cot(full_output)
		else:
			steps, answer = "", full_output

		return {
			"question": question,
			"steps": steps,
			"answer": answer,
			"full_output": full_output
		}

	def _parse_cot(self, text):
		"""解析推理链和最终答案（需根据实际数据格式调整）"""
		steps = []
		answer = ""
		lines = text.split("\n")
		for line in lines:
			if line.startswith("步骤"):
				steps.append(line)
			elif line.startswith("答案"):
				answer = line
		return "\n".join(steps), answer


# --------------------- 损失函数 ---------------------
class CoTLoss(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.answer_criterion = nn.CrossEntropyLoss(ignore_index=-100)
		self.step_criterion = nn.CrossEntropyLoss(ignore_index=-100)
		self.answer_weight = config.answer_loss_weight
		self.step_weight = config.step_loss_weight
		self.length_penalty = config.length_penalty

	def forward(self, student_outputs, teacher_data):
		# 答案损失
		answer_logits = student_outputs.logits
		answer_labels = self._prepare_labels(teacher_data["answer"])
		answer_loss = self.answer_criterion(
			answer_logits.view(-1, answer_logits.size(-1)),
			answer_labels.view(-1)
		)

		# 推理步骤损失
		step_logits = student_outputs.step_logits  # 假设学生模型有步骤输出头
		step_labels = self._prepare_labels(teacher_data["steps"])
		step_loss = self.step_criterion(
			step_logits.view(-1, step_logits.size(-1)),
			step_labels.view(-1)
		)

		# 长度正则化
		avg_length = student_outputs.lengths.float().mean()
		length_loss = self.length_penalty * avg_length

		total_loss = (
				self.answer_weight * answer_loss +
				self.step_weight * step_loss +
				length_loss
		)

		return {
			"total_loss": total_loss,
			"answer_loss": answer_loss,
			"step_loss": step_loss,
			"length_loss": length_loss
		}

	def _prepare_labels(self, text_batch):
		"""将文本转换为标签（示例实现）"""
		# 实际需根据tokenizer实现
		return torch.tensor([[0, 1, ...]], dtype=torch.long)

# --------------------- 训练循环 ---------------------


def train_epoch(student_model, dataloader, optimizer, scheduler, scaler, loss_fn, config):
	student_model.train()
	total_loss = 0.0

	for batch in tqdm(dataloader, desc="Training"):
		# 前向传播
		with torch.cuda.amp.autocast(enabled=config.fp16):
			outputs = student_model(
				input_ids=batch["input_ids"].to(config.device),
				attention_mask=batch["attention_mask"].to(config.device)
			)

			loss_dict = loss_fn(outputs, batch)

		# 反向传播
		scaler.scale(loss_dict["total_loss"]).backward()

		# 梯度裁剪
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

		# 参数更新
		scaler.step(optimizer)
		scaler.update()
		optimizer.zero_grad()
		scheduler.step()

		total_loss += loss_dict["total_loss"].item()

	return total_loss / len(dataloader)


# --------------------- 主函数 ---------------------
def main():
	config = TrainingConfig()
	config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 初始化教师模型
	teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
	teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model)
	teacher_model.eval().to(config.device)

	# 初始化学生模型
	student_tokenizer = AutoTokenizer.from_pretrained(config.student_model)
	student_model = AutoModelForCausalLM.from_pretrained(config.student_model)
	if config.num_gpus > 1:
		student_model = nn.DataParallel(student_model)
	student_model.to(config.device)

	# 准备数据（示例问题列表）
	train_questions = [
		"解方程2x^2 - 8 = 0。",
		"计算从1加到100的和。",
		# ... 添加更多问题
	]

	dataset = CoTDataset(train_questions, teacher_tokenizer, teacher_model, config)
	dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

	# 优化器与调度器
	optimizer = AdamW(student_model.parameters(), lr=config.learning_rate)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=100,
		num_training_steps=len(dataloader) * config.num_epochs
	)

	# 混合精度训练
	scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

	# 损失函数
	loss_fn = CoTLoss(config)

	# 训练循环
	for epoch in range(config.num_epochs):
		avg_loss = train_epoch(
			student_model, dataloader,
			optimizer, scheduler, scaler, loss_fn, config
		)
		print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

	# 保存学生模型
	student_model.save_pretrained("./distilled_model")
	student_tokenizer.save_pretrained("./distilled_model")


if __name__ == "__main__":
	main()