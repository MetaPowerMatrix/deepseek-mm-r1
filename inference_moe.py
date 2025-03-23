import torch
import torch.nn.functional as F
import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from simple_moe import (
    SimpleMoE, 
    TransformerMoE, 
    LongContextTransformerMoE,
    count_parameters
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class InferenceConfig:
    """推理配置"""
    
    def __init__(self):
        # 模型设置
        self.model_type = "transformer_moe"  # 可选: simple_moe, transformer_moe, long_transformer_moe
        self.model_path = "./checkpoints/model.pt"
        self.vocab_path = "./data/tokenizer.json"
        
        # 模型参数 (用于构建模型)
        self.vocab_size = 30000
        self.d_model = 768
        self.num_heads = 12
        self.num_layers = 6
        self.d_ff = 2048
        self.max_seq_len = 2048
        self.num_experts = 4
        self.k = 2
        
        # 用于SimpleMoE
        self.input_size = 1024
        self.hidden_size = 1024
        self.output_size = 1024
        
        # 用于LongContextTransformerMoE
        self.scaling_factor = 8.0
        self.rope_theta = 10000
        
        # 生成设置
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.1
        
        # 硬件设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(vocab_path):
    """从文件加载分词器，如果文件不存在则创建默认分词器"""
    try:
        from dataset_moe import Tokenizer
        if os.path.exists(vocab_path):
            return Tokenizer.from_file(vocab_path)
        else:
            logging.warning(f"找不到词表文件: {vocab_path}，创建默认分词器")
            return Tokenizer.create_default(save_path=vocab_path)
    except ImportError:
        logging.warning("未找到dataset_moe模块，使用简单的字符级分词器")
        
        class SimpleTokenizer:
            def __init__(self, vocab_size=30000):
                self.vocab_size = vocab_size
                # 创建一个简单字符映射
                chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:-_()[]{}'\"\n ")
                self.token_to_id = {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<bos>": 2,
                    "<eos>": 3,
                }
                
                # 添加字符到词表
                for i, char in enumerate(chars):
                    self.token_to_id[char] = i + 4
                
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            
            def encode(self, text, add_special_tokens=False):
                if add_special_tokens:
                    tokens = [2]  # <bos>
                else:
                    tokens = []
                
                for char in text:
                    tokens.append(self.token_to_id.get(char, 1))  # 未知字符用<unk>
                
                if add_special_tokens:
                    tokens.append(3)  # <eos>
                
                return tokens
            
            def decode(self, ids, skip_special_tokens=True):
                if skip_special_tokens:
                    ids = [id for id in ids if id >= 4]  # 跳过特殊token
                
                return "".join([self.id_to_token.get(id, "<unk>") for id in ids])
        
        return SimpleTokenizer(vocab_size=30000)


def load_model(config):
    """根据配置加载模型"""
    try:
        if config.model_type == "simple_moe":
            model = SimpleMoE(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                output_size=config.output_size,
                num_experts=config.num_experts,
                k=config.k
            )
        elif config.model_type == "transformer_moe":
            model = TransformerMoE(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                num_experts=config.num_experts,
                k=config.k
            )
        elif config.model_type == "long_transformer_moe":
            model = LongContextTransformerMoE(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                num_experts=config.num_experts,
                k=config.k,
                scaling_factor=config.scaling_factor,
                rope_theta=config.rope_theta
            )
        else:
            raise ValueError(f"不支持的模型类型: {config.model_type}")
        
        # 加载预训练权重（如果存在）
        if os.path.exists(config.model_path):
            # 加载checkpoint
            checkpoint = torch.load(config.model_path, map_location=config.device)
            
            # 检查checkpoint中是否包含模型状态字典
            if "model_state_dict" in checkpoint:
                # checkpoint包含完整训练状态
                model.load_state_dict(checkpoint["model_state_dict"])
                logging.info(f"从{config.model_path}加载模型权重 (从checkpoint)")
                
                # 如果checkpoint中包含词表路径，可以更新配置
                if "vocab_path" in checkpoint and not os.path.exists(config.vocab_path):
                    saved_vocab_path = checkpoint["vocab_path"]
                    if os.path.exists(saved_vocab_path):
                        config.vocab_path = saved_vocab_path
                        logging.info(f"使用checkpoint中记录的词表路径: {saved_vocab_path}")
                
                # 如果checkpoint中包含配置信息，可以更新模型配置
                if "config" in checkpoint:
                    saved_config = checkpoint["config"]
                    for key, value in saved_config.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    logging.info("从checkpoint更新模型配置")
            else:
                # 直接加载模型状态字典
                model.load_state_dict(checkpoint)
                logging.info(f"从{config.model_path}加载模型权重")
        else:
            logging.warning(f"找不到模型权重文件: {config.model_path}，使用随机初始化")
        
        model = model.to(config.device)
        model.eval()  # 设置为评估模式
        
        logging.info(f"模型参数量: {count_parameters(model):.2f}M")
        return model
    
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise


def generate_simple_moe(model, input_vector, config):
    """
    使用SimpleMoE模型生成输出
    
    参数:
        model: SimpleMoE模型
        input_vector: 输入向量 [batch_size, input_size]
        config: 配置
    
    返回:
        生成的向量 [batch_size, output_size]
    """
    with torch.no_grad():
        output = model(input_vector)
    return output


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    对logits应用top-k和top-p过滤
    
    参数:
        logits: 词汇表上的logits
        top_k: 保留概率最高的top_k个token
        top_p: 累积概率达到top_p的token将被保留
        filter_value: 被过滤token的logit值
    
    返回:
        过滤后的logits
    """
    top_k = min(top_k, logits.size(-1))  # 安全检查
    
    if top_k > 0:
        # 移除概率最小的所有token，仅保留top_k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个token（概率最高的）保留
        sorted_indices_to_remove[..., 0] = 0
        
        # 回到原始索引空间
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits


def apply_repetition_penalty(logits, input_ids, repetition_penalty):
    """
    应用重复惩罚，降低已生成token的概率
    
    参数:
        logits: 当前step的logits
        input_ids: 已生成的token ids
        repetition_penalty: 重复惩罚系数
    
    返回:
        应用惩罚后的logits
    """
    for prev_token in set(input_ids.tolist()):
        # 如果token在之前出现过，降低其概率
        logits[prev_token] /= repetition_penalty
    return logits


def encode_safe(tokenizer, text, add_special_tokens=True):
    """安全的编码方法，处理可能出现的错误"""
    try:
        # 直接使用tokenizer.encode
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except Exception as e:
        logging.warning(f"使用tokenizer.encode失败: {str(e)}，回退到基本字符编码")
        # 回退方案：基本的字符级处理
        if not hasattr(tokenizer, 'token_to_id'):
            # 创建一个极简单的映射
            special_tokens = {
                "<pad>": 0,
                "<unk>": 1, 
                "<bos>": 2,
                "<eos>": 3
            }
            token_to_id = {char: i+10 for i, char in enumerate(text)}
            for k, v in special_tokens.items():
                token_to_id[k] = v
            tokenizer.token_to_id = token_to_id
        
        result = []
        if add_special_tokens:
            result.append(tokenizer.token_to_id.get("<bos>", 2))
            
        for char in text:
            result.append(tokenizer.token_to_id.get(char, tokenizer.token_to_id.get("<unk>", 1)))
            
        if add_special_tokens:
            result.append(tokenizer.token_to_id.get("<eos>", 3))
            
        return result


def decode_safe(tokenizer, ids, skip_special_tokens=True):
    """安全的解码方法，处理可能出现的错误"""
    try:
        # 直接使用tokenizer.decode
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    except Exception as e:
        logging.warning(f"使用tokenizer.decode失败: {str(e)}，回退到基本字符解码")
        # 回退方案：基本的id到字符映射
        if not hasattr(tokenizer, 'id_to_token'):
            # 如果没有id_to_token，但有token_to_id，就反转它
            if hasattr(tokenizer, 'token_to_id'):
                tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
            else:
                # 极端情况，返回id
                return " ".join([str(i) for i in ids])
        
        if skip_special_tokens:
            # 跳过特殊token，假设id < 10是特殊token
            ids = [i for i in ids if i >= 10]
            
        # 转换为字符并连接
        return "".join([tokenizer.id_to_token.get(i, "<unk>") for i in ids])


def generate_text(model, tokenizer, prompt, config):
    """
    使用Transformer模型生成文本
    
    参数:
        model: TransformerMoE或LongContextTransformerMoE模型
        tokenizer: 分词器
        prompt: 提示文本
        config: 配置
    
    返回:
        生成的文本
    """
    # 安全编码提示文本
    input_ids = torch.tensor(
        encode_safe(tokenizer, prompt, add_special_tokens=True),
        dtype=torch.long,
        device=config.device
    ).unsqueeze(0)  # 添加batch维度
    
    # 准备生成
    generated_tokens = []
    past_length = input_ids.size(1)
    max_length = past_length + config.max_new_tokens
    
    try:
        with torch.no_grad():
            for _ in tqdm(range(config.max_new_tokens), desc="生成中"):
                # 确保不超过模型的最大序列长度
                if input_ids.size(1) > config.max_seq_len:
                    input_ids = input_ids[:, -config.max_seq_len:]
                
                # 前向传播
                outputs = model(input_ids)  # [batch_size, seq_len, vocab_size]
                
                # 获取最后一个token的logits
                next_token_logits = outputs[0, -1, :]
                
                # 应用重复惩罚
                if config.repetition_penalty > 1.0:
                    next_token_logits = apply_repetition_penalty(
                        next_token_logits, input_ids[0], config.repetition_penalty
                    )
                
                # 应用温度系数
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # 应用top-k和top-p过滤
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, 
                    top_k=config.top_k, 
                    top_p=config.top_p
                )
                
                # 采样下一个token
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 检查是否生成了EOS（结束）
                eos_id = getattr(tokenizer, 'special_tokens', {}).get('<eos>', 3)
                if next_token.item() == eos_id:
                    break
                
                # 添加到已生成的token
                generated_tokens.append(next_token.item())
                
                # 更新输入序列
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    except Exception as e:
        logging.error(f"生成过程中出错: {str(e)}")
        return prompt + "[生成失败]"
    
    # 安全解码生成的token
    generated_text = decode_safe(tokenizer, generated_tokens)
    return prompt + generated_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MoE模型推理脚本")
    parser.add_argument("--model_type", type=str, default="transformer_moe",
                        choices=["simple_moe", "transformer_moe", "long_transformer_moe"],
                        help="模型类型")
    parser.add_argument("--model_path", type=str, default="./checkpoints/model.pt",
                        help="模型权重路径")
    parser.add_argument("--vocab_path", type=str, default="./data/tokenizer.json",
                        help="词表文件路径")
    parser.add_argument("--prompt", type=str, default="",
                        help="生成的提示文本")
    parser.add_argument("--input_file", type=str, default="",
                        help="输入文件，包含多行文本作为提示")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="最大生成token数量")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p采样阈值")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k采样阈值")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚系数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建配置
    config = InferenceConfig()
    config.model_type = args.model_type
    config.model_path = args.model_path
    config.vocab_path = args.vocab_path
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    config.repetition_penalty = args.repetition_penalty
    
    # 加载模型和分词器
    model = load_model(config)
    tokenizer = load_tokenizer(config.vocab_path)
    
    if args.model_type == "simple_moe":
        # 生成随机输入向量进行测试
        input_vector = torch.randn(1, config.input_size, device=config.device)
        output = generate_simple_moe(model, input_vector, config)
        print(f"SimpleMoE输出向量形状: {output.shape}")
        
        # 输出前10个值
        print(f"输出向量前10个值: {output[0, :10].cpu().numpy()}")
    else:
        if args.input_file:
            # 从文件加载多个提示
            with open(args.input_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            for i, prompt in enumerate(prompts):
                print(f"\n==== 提示 {i+1}/{len(prompts)} ====")
                print(f"输入: {prompt}")
                generated = generate_text(model, tokenizer, prompt, config)
                print(f"输出: {generated}")
                print("="*50)
        else:
            # 使用单个提示
            prompt = args.prompt if args.prompt else "这是一个测试："
            print(f"输入提示: {prompt}")
            generated = generate_text(model, tokenizer, prompt, config)
            print(f"生成文本: {generated}")


if __name__ == "__main__":
    main() 