# dataset.py
import torch
import os
from collections import Counter
from transformers import AutoTokenizer


class TextDataset(torch.utils.data.Dataset):
	def __init__(self, directory_path, seq_length, tokenizer):
		self.seq_length = seq_length
		self.tokenizer = tokenizer
		self.data = []
		self.vocab = {}
		self.inverse_vocab = {}

		# 第一步：统计所有单词的频率
		word_counter = Counter()

		# 遍历 directory_path 目录中的所有 .tokenized.txt 文件
		for filename in os.listdir(directory_path):
			if filename.endswith(".tokenized.txt"):
				file_path = os.path.join(directory_path, filename)
				with open(file_path, "r", encoding="utf-8") as f:
					words = f.read().split()
					word_counter.update(words)

		# 第二步：创建词汇表，给每个单词分配一个 ID
		self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counter.items())}
		self.vocab['<pad>'] = 0  # 为 padding 添加一个 ID
		self.vocab['<unk>'] = len(self.vocab)  # 为未知单词添加一个 ID

		# 创建逆词汇表
		self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}

		# 第三步：将文本转换为 token ID
		for filename in os.listdir(directory_path):
			if filename.endswith(".tokenized.txt"):
				file_path = os.path.join(directory_path, filename)
				with open(file_path, "r", encoding="utf-8") as f:
					words = f.read().split()
					# 将每个单词转换为 token ID，如果不在词汇表中则使用 <unk>
					token_ids = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
					self.data.append(token_ids)

		# 将数据转化为训练所需的序列形式
		self.data = [self.pad_sequence(seq) for seq in self.data]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		input_text = self.data[idx]

		# 编码输入文本
		input_ids = torch.tensor(input_text)  # 转换为 tensor
		target_ids = input_ids.clone()  # 使用输入作为目标
		return input_ids, target_ids

	def pad_sequence(self, seq):
		"""填充序列到 seq_length"""
		if len(seq) < self.seq_length:
			# 使用 pad token 填充
			seq += [self.vocab['<pad>']] * (self.seq_length - len(seq))
		else:
			# 如果超出 seq_length，则截断
			seq = seq[:self.seq_length]
		return seq

	'''
	def __getitem__(self, idx):
		input_ids = torch.tensor(self.data[idx], dtype=torch.long)

		# 如果输入序列长度小于 seq_length，进行填充
		padding_length = self.seq_length - input_ids.size(0)
		if padding_length > 0:
			padding = torch.tensor([self.vocab['<pad>']] * padding_length, dtype=torch.long)
			input_ids = torch.cat([input_ids, padding], dim=0)

		# 设置 target_ids 为 input_ids 的下一个 token（即语言模型的训练目标）
		target_ids = input_ids[1:].clone()
		target_ids = torch.cat([target_ids, torch.tensor([self.vocab['<pad>']], dtype=torch.long)])

		return input_ids, target_ids
'''