import spacy
import io
import argparse
import glob
import os
import tqdm
from multiprocessing import Pool
from functools import partial
import chardet

def detect_encoding(file_path):
    """检测文件的实际编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024)  # 读取文件的前 1KB 数据
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'  # 如果检测失败，默认返回 'utf-8'



def save_tokenized_text(output_dir, filename, text):
    # 构建完整输出路径
    text_file = os.path.join(output_dir, filename)

    # 确保目标目录存在
    os.makedirs(os.path.dirname(text_file), exist_ok=True)

    # 保存文件
    with io.open(text_file, 'w', encoding='utf-8') as fo:
        fo.write(text)

def tokenizeSpacy(args):
    nlp = spacy.load("en_core_web_sm")  # 加载 spaCy 模型
    extraction_file_paths = glob.glob(args.input_glob)

    for extraction_file_path in extraction_file_paths:
        path, filename = os.path.split(extraction_file_path)
        text_file = os.path.join(
            args.output_dir, filename.replace('.txt', '.tokenized.txt'))

        # 确保输出目录存在
        os.makedirs(os.path.dirname(text_file), exist_ok=True)

        # 检测文件编码
        file_encoding = detect_encoding(extraction_file_path)

        try:
            # 打开输入文件和输出文件
            with io.open(extraction_file_path, 'r', encoding=file_encoding) as fi, \
                    io.open(text_file, 'w', encoding='utf-8') as fo:

                omitted_line_count = 0
                for line in fi:
                    if len(line.strip()) > 0:  # 忽略空行
                        doc = nlp(line)
                        fo.write(' '.join([x.text for x in doc]) + '\n')
                    else:
                        omitted_line_count += 1

            print(f'Omitted {omitted_line_count} empty lines from {filename}')
        except UnicodeDecodeError:
            print(f"Failed to decode {extraction_file_path} with encoding {file_encoding}. Skipping this file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str, default='*.txt')
    parser.add_argument('--output_dir', type=str, default='tokenized')
    parser.add_argument('--tokenizer', type=str, default='spacy', choices=['spacy', 'gpt2'])
    parser.add_argument('--combine', type=int, default=1e8, help="min tokens per file in gpt2 mode")
    parser.add_argument('--file_bs', type=int, default=10000, help="files per batch in gpt2 mode")

    # 解析命令行参数
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 根据 tokenizer 选择执行的函数
    if args.tokenizer == 'spacy':
        tokenizeSpacy(args)
    else:
        print("GPT-2 tokenizer is not implemented in this version.")