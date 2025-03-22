"""
混合专家模型(Mixture of Experts)数据集模块
提供数据加载和预处理功能
"""

import os
import logging
import torch
import numpy as np
import json
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class Tokenizer:
    """简单的分词器实现，用于TransformerMoE模型"""
    
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.token_to_id = {}  # 词到ID的映射
        self.id_to_token = {}  # ID到词的映射
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # 初始化特殊词表
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.vocab_initialized = False
        
    def build_vocab(self, texts, min_freq=2):
        """
        从文本列表构建词表
        
        参数:
            texts: 文本列表
            min_freq: 最小词频
        """
        logging.info("开始构建词表...")
        
        # 统计词频
        counter = Counter()
        for text in texts:
            words = self._tokenize(text)
            counter.update(words)
        
        # 过滤低频词并按频率排序
        vocab = [token for token, count in counter.most_common() 
                if count >= min_freq]
        
        # 限制词表大小
        vocab = vocab[:self.vocab_size - len(self.special_tokens)]
        
        # 构建映射
        for idx, token in enumerate(vocab, start=len(self.special_tokens)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.vocab_initialized = True
        logging.info(f"词表构建完成，大小: {len(self.token_to_id)}")
        
    def _tokenize(self, text):
        """简单的分词方法，可以根据需要扩展"""
        # 这里简单地按字符分词，实际应用中应该使用更复杂的分词算法
        return list(text)
    
    def encode(self, text, add_special_tokens=False):
        """将文本转换为ID序列"""
        if not self.vocab_initialized:
            raise ValueError("词表尚未初始化，请先调用build_vocab方法")
        
        tokens = self._tokenize(text)
        
        if add_special_tokens:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        
        # 将词转换为ID，对于未知词使用<unk>的ID
        ids = [self.token_to_id.get(token, self.special_tokens["<unk>"]) 
               for token in tokens]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """将ID序列转换为文本"""
        if not self.vocab_initialized:
            raise ValueError("词表尚未初始化，请先调用build_vocab方法")
        
        # 过滤特殊词
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in self.special_tokens.values()]
        
        # 将ID转换为词
        tokens = [self.id_to_token.get(idx, "<unk>") for idx in ids]
        
        # 拼接成文本
        text = "".join(tokens)
        
        return text
    
    def save_vocab(self, vocab_path):
        """保存词表到文件"""
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        vocab_data = {
            "token_to_id": self.token_to_id,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"词表已保存到: {vocab_path}")
    
    @classmethod
    def from_file(cls, vocab_path):
        """从文件加载词表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 如果词表中没有vocab_size字段，根据token_to_id的大小估算
        if "vocab_size" not in vocab_data:
            vocab_size = max(30000, len(vocab_data.get("token_to_id", {})) * 2)
            logging.warning(f"词表文件缺少vocab_size字段，使用默认值: {vocab_size}")
        else:
            vocab_size = vocab_data["vocab_size"]
        
        tokenizer = cls(vocab_size=vocab_size)
        
        # 确保token_to_id字段存在
        if "token_to_id" not in vocab_data:
            raise ValueError(f"词表文件{vocab_path}格式错误: 缺少token_to_id字段")
            
        tokenizer.token_to_id = vocab_data["token_to_id"]
        
        # 处理特殊token
        if "special_tokens" in vocab_data:
            tokenizer.special_tokens = vocab_data["special_tokens"]
        else:
            logging.warning(f"词表文件缺少special_tokens字段，使用默认值")
        
        # 重建id_to_token映射
        tokenizer.id_to_token = {int(id): token for token, id in tokenizer.token_to_id.items()}
        tokenizer.vocab_initialized = True
        
        logging.info(f"从{vocab_path}加载词表，大小: {len(tokenizer.token_to_id)}")
        return tokenizer
    
    @classmethod
    def create_default(cls, save_path=None):
        """创建默认词表"""
        # 首先尝试加载已有的tokenizer.json文件
        tokenizer_path = "./data/tokenizer.json"
        if os.path.exists(tokenizer_path):
            try:
                logging.info(f"尝试从{tokenizer_path}加载词表")
                return cls.from_file(tokenizer_path)
            except Exception as e:
                logging.warning(f"从{tokenizer_path}加载词表失败: {str(e)}")
        
        vocab_size = 30000
        tokenizer = cls(vocab_size=vocab_size)
        
        # 为了测试，我们创建一个简单的默认词表
        # 包含数字、字母和一些标点符号
        default_chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:-_()[]{}'\"\n ")
        
        # 构建映射
        for idx, token in enumerate(default_chars, start=len(tokenizer.special_tokens)):
            if idx < vocab_size:
                tokenizer.token_to_id[token] = idx
                tokenizer.id_to_token[idx] = token
        
        tokenizer.vocab_initialized = True
        logging.info(f"创建默认词表，大小: {len(tokenizer.token_to_id)}")
        
        if save_path:
            tokenizer.save_vocab(save_path)
            
        return tokenizer


class MoEDataset(Dataset):
    """用于MoE模型的数据集类"""
    
    def __init__(self, file_path, max_seq_length=None, model_type="simple_moe", tokenizer=None):
        """
        初始化MoE数据集
        
        参数:
            file_path: 数据集文件路径
            max_seq_length: 最大序列长度(仅TransformerMoE需要)
            model_type: 模型类型，"simple_moe"或"transformer_moe"
            tokenizer: 分词器对象，用于TransformerMoE模型
        """
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
        # 如果是transformer模型但没有提供tokenizer，创建默认的
        if model_type == "transformer_moe" and tokenizer is None:
            # 首先尝试直接使用data/tokenizer.json
            tokenizer_path = "./data/tokenizer.json"
            if os.path.exists(tokenizer_path):
                try:
                    logging.info(f"尝试从{tokenizer_path}加载词表...")
                    self.tokenizer = Tokenizer.from_file(tokenizer_path)
                    logging.info(f"成功加载词表，大小: {len(self.tokenizer.token_to_id)}")
                except Exception as e:
                    logging.warning(f"从{tokenizer_path}加载词表失败: {str(e)}")
                    # 回退到vocab.json
                    vocab_dir = os.path.dirname(file_path)
                    vocab_path = os.path.join(vocab_dir, "vocab.json")
                    
                    if os.path.exists(vocab_path):
                        self.tokenizer = Tokenizer.from_file(vocab_path)
                    else:
                        self.tokenizer = Tokenizer.create_default(save_path=vocab_path)
            else:
                # 回退到vocab.json
                vocab_dir = os.path.dirname(file_path)
                vocab_path = os.path.join(vocab_dir, "vocab.json")
                
                if os.path.exists(vocab_path):
                    self.tokenizer = Tokenizer.from_file(vocab_path)
                else:
                    self.tokenizer = Tokenizer.create_default(save_path=vocab_path)
        
        if not os.path.exists(file_path):
            # 如果没有真实数据集，创建合成数据集
            self.create_synthetic_dataset(file_path, max_seq_length, model_type)
        
        # 加载数据集
        self.data = torch.load(file_path)
        logging.info(f"加载数据集: {file_path}, 样本数: {len(self.data)}")
        
    def create_synthetic_dataset(self, file_path, max_seq_length, model_type, num_samples=1000):
        """
        创建合成数据集用于训练
        
        参数:
            file_path: 保存数据集的路径
            max_seq_length: 最大序列长度
            model_type: 模型类型
            num_samples: 样本数量
        """
        logging.info(f"未找到数据集，创建合成数据集: {model_type}...")
        
        if model_type == "simple_moe":
            # 为SimpleMoE创建向量数据
            input_size = 1024
            data = torch.randn(num_samples, input_size)  # 随机输入特征
            labels = torch.randn(num_samples, input_size)  # 随机标签
            
            self.data = [(x, y) for x, y in zip(data, labels)]
            
        else:  # transformer_moe
            # 为TransformerMoE创建序列数据
            vocab_size = len(self.tokenizer.token_to_id) if self.tokenizer else 30000
            if not max_seq_length:
                max_seq_length = 512  # 默认序列长度
                
            data = torch.randint(0, vocab_size, (num_samples, max_seq_length))  # 随机token序列
            labels = data.clone()  # 自回归任务中，输入即标签
            
            self.data = [(x, y) for x, y in zip(data, labels)]
        
        # 创建输出目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存数据集
        torch.save(self.data, file_path)
        logging.info(f"创建合成数据集完成: {file_path}, 样本数: {num_samples}")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        return self.data[idx]
    
    @staticmethod
    def create_from_text(file_path, output_path, tokenizer=None, block_size=512):
        """
        从文本文件创建TransformerMoE的数据集
        
        参数:
            file_path: 输入文本文件路径
            output_path: 输出数据集路径
            tokenizer: 分词器，如果为None则创建默认分词器
            block_size: 文本块大小
        """
        examples = []
        
        try:
            # 如果没有提供tokenizer，创建默认的
            if tokenizer is None:
                vocab_dir = os.path.dirname(output_path)
                vocab_path = os.path.join(vocab_dir, "vocab.json")
                
                if os.path.exists(vocab_path):
                    tokenizer = Tokenizer.from_file(vocab_path)
                else:
                    # 从文件构建词表
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    tokenizer = Tokenizer(vocab_size=30000)
                    tokenizer.build_vocab([text])
                    tokenizer.save_vocab(vocab_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # 将文本分块为固定长度序列
            tokenized_text = tokenizer.encode(text)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                input_ids = torch.tensor(tokenized_text[i:i + block_size], dtype=torch.long)
                examples.append((input_ids, input_ids.clone()))
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据集
            torch.save(examples, output_path)
            logging.info(f"从文本创建数据集完成: {output_path}, 样本数: {len(examples)}")
            
            return len(examples)
        
        except Exception as e:
            logging.error(f"创建数据集失败: {str(e)}")
            return 0
    
    @staticmethod
    def create_from_vectors(vectors, labels, output_path):
        """
        从特征向量创建SimpleMoE的数据集
        
        参数:
            vectors: 输入特征向量 [N, input_size]
            labels: 目标向量 [N, output_size]
            output_path: 输出数据集路径
        """
        try:
            if isinstance(vectors, np.ndarray):
                vectors = torch.from_numpy(vectors).float()
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).float()
                
            examples = [(x, y) for x, y in zip(vectors, labels)]
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据集
            torch.save(examples, output_path)
            logging.info(f"从向量创建数据集完成: {output_path}, 样本数: {len(examples)}")
            
            return len(examples)
        
        except Exception as e:
            logging.error(f"创建数据集失败: {str(e)}")
            return 0

    @staticmethod
    def create_from_jsonl(jsonl_path, output_path, tokenizer=None, max_seq_length=512, limit=None):
        """
        从JSONL文件创建TransformerMoE的数据集
        
        参数:
            jsonl_path: JSONL文件路径
            output_path: 输出数据集路径
            tokenizer: 分词器，如果为None则创建默认分词器
            max_seq_length: 最大序列长度
            limit: 限制处理的样本数量，None表示处理所有样本
        
        返回:
            处理的样本数量
        """
        import json
        from tqdm import tqdm
        
        try:
            # 如果没有提供tokenizer，创建默认的
            if tokenizer is None:
                logging.info("未提供tokenizer，尝试加载或创建默认tokenizer")
                # 首先尝试加载tokenizer.json
                tokenizer_path = "./data/tokenizer.json"
                if os.path.exists(tokenizer_path):
                    try:
                        tokenizer = Tokenizer.from_file(tokenizer_path)
                    except Exception as e:
                        logging.warning(f"从{tokenizer_path}加载词表失败: {str(e)}")
                        # 创建默认tokenizer
                        tokenizer = Tokenizer.create_default(save_path=tokenizer_path)
                else:
                    # 创建默认tokenizer
                    tokenizer = Tokenizer.create_default(save_path=tokenizer_path)
            
            examples = []
            line_count = 0
            
            # 统计文件行数用于进度条
            if limit is None:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        line_count += 1
                total_lines = line_count
            else:
                total_lines = min(limit, line_count) if line_count > 0 else limit
            
            logging.info(f"开始处理JSONL文件: {jsonl_path}")
            processed_count = 0
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f), total=total_lines, desc="处理JSONL"):
                    if limit is not None and i >= limit:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取文本字段
                        # 根据观察到的JSONL格式，假设有instruction, input和output字段
                        instruction = data.get("instruction", "")
                        input_text = data.get("input", "")
                        output_text = data.get("output", "")
                        
                        # 组合文本，格式: instruction [SEP] input [SEP] output
                        combined_text = ""
                        if instruction:
                            combined_text += instruction
                        if input_text:
                            if combined_text:
                                combined_text += " "
                            combined_text += input_text
                        
                        # 编码文本
                        input_ids = tokenizer.encode(combined_text, add_special_tokens=True)
                        target_ids = tokenizer.encode(output_text, add_special_tokens=True)
                        
                        # 截断或填充到max_seq_length
                        if len(input_ids) > max_seq_length:
                            input_ids = input_ids[:max_seq_length]
                        else:
                            # 填充
                            input_ids = input_ids + [tokenizer.special_tokens["<pad>"]] * (max_seq_length - len(input_ids))
                            
                        if len(target_ids) > max_seq_length:
                            target_ids = target_ids[:max_seq_length]
                        else:
                            # 填充
                            target_ids = target_ids + [tokenizer.special_tokens["<pad>"]] * (max_seq_length - len(target_ids))
                        
                        # 转换为tensor
                        input_tensor = torch.tensor(input_ids, dtype=torch.long)
                        target_tensor = torch.tensor(target_ids, dtype=torch.long)
                        
                        examples.append((input_tensor, target_tensor))
                        processed_count += 1
                        
                    except json.JSONDecodeError:
                        logging.warning(f"无法解析第{i+1}行JSON: {line[:50]}...")
                    except Exception as e:
                        logging.warning(f"处理第{i+1}行时出错: {str(e)}")
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据集
            torch.save(examples, output_path)
            logging.info(f"从JSONL创建数据集完成: {output_path}, 样本数: {processed_count}")
            
            return processed_count
        
        except Exception as e:
            logging.error(f"创建数据集失败: {str(e)}")
            return 0


def create_dataloader(dataset, batch_size=8, num_gpus=1, shuffle=True, num_workers=4):
    """
    创建数据加载器
    
    参数:
        dataset: MoEDataset实例
        batch_size: 批次大小(单GPU)
        num_gpus: GPU数量
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
    
    返回:
        DataLoader实例
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        shuffle=shuffle
    ) if num_gpus > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 测试分词器
    tokenizer = Tokenizer.create_default(save_path="./data/vocab.json")
    sample_text = "Hello, this is a test for the MoE model."
    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(tokens)
    logging.info(f"原始文本: {sample_text}")
    logging.info(f"编码后: {tokens}")
    logging.info(f"解码后: {decoded_text}")
    
    # 测试SimpleMoE数据集
    simple_dataset = MoEDataset(
        file_path="./data/simple_moe_test.pt",
        model_type="simple_moe"
    )
    
    # 测试TransformerMoE数据集
    transformer_dataset = MoEDataset(
        file_path="./data/transformer_moe_test.pt",
        max_seq_length=128,
        model_type="transformer_moe",
        tokenizer=tokenizer
    )
    
    # 测试数据加载器
    simple_loader = create_dataloader(simple_dataset, batch_size=4)
    transformer_loader = create_dataloader(transformer_dataset, batch_size=2)
    
    # 输出数据集信息
    logging.info(f"SimpleMoE 数据集大小: {len(simple_dataset)}")
    logging.info(f"TransformerMoE 数据集大小: {len(transformer_dataset)}")
    
    # 检查数据样本
    simple_batch = next(iter(simple_loader))
    transformer_batch = next(iter(transformer_loader))
    
    logging.info(f"SimpleMoE 批次形状: {simple_batch[0].shape}, {simple_batch[1].shape}")
    logging.info(f"TransformerMoE 批次形状: {transformer_batch[0].shape}, {transformer_batch[1].shape}")
    
    # 测试从文本创建数据集
    sample_text_path = "./data/sample_text.txt"
    with open(sample_text_path, "w", encoding="utf-8") as f:
        f.write("这是一个测试文本，用于创建TransformerMoE数据集。" * 100)
    
    num_samples = MoEDataset.create_from_text(
        file_path=sample_text_path,
        output_path="./data/text_dataset.pt",
        tokenizer=tokenizer,
        block_size=64
    )
    logging.info(f"从文本创建的数据集样本数: {num_samples}") 