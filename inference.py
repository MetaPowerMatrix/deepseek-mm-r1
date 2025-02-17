# inference.py
import torch
import sys
import os
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Adding to sys.path:", project_root)
sys.path.append(project_root)

from model.gpt import GPT
from transformers import AutoTokenizer


def load_model():
	# 加载模型
	model = GPT(vocab_size=50000, embed_size=256, num_layers=6, num_heads=8, max_length=512)
	model.load_state_dict(torch.load("trained_model/model.pth"), strict=False)
	model.eval()  # 设置为评估模式

	# 加载分词器
	tokenizer = AutoTokenizer.from_pretrained('gpt2')

	return model, tokenizer  # 返回模型和分词器


def chat():
	model, tokenizer = load_model()  # 加载模型和分词器

	while True:
		text = input("Input: ")  # 获取用户输入
		if text.lower() == "quit":
			break

		# 对输入文本进行编码
		input_ids = tokenizer.encode(text, return_tensors='pt')

		# 生成文本
		generated_ids = model.generate(input_ids, max_length=100, temperature=1.0, top_k=50)

		# 解码并打印生成的文本
		output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
		print(f"GPT: {output_text}")


if __name__ == "__main__":
	chat()