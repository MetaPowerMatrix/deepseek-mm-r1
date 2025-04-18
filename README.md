# SimpleMoE

SimpleMoE 是一个简洁的混合专家模型（Mixture of Experts, MoE）实现，包含了不同类型的MoE模型结构和推理生成脚本。

## 项目结构

- `simple_moe.py`: MoE模型的核心实现，包含以下主要模型：
  - `SimpleMoE`: 简单的前馈网络MoE模型
  - `TransformerMoE`: 基于Transformer架构的MoE模型
  - `LongContextTransformerMoE`: 支持超长序列的Transformer MoE模型
  
- `rotary_embeddings.py`: 旋转位置编码（RoPE）的实现，包含：
  - `RotaryEmbedding`: 基础RoPE实现
  - `LinearScalingRotaryEmbedding`: 线性缩放RoPE
  - `DynamicNTKScalingRotaryEmbedding`: 动态NTK缩放RoPE
  - `YarnRotaryEmbedding`: 基于YaRN技术的RoPE长序列扩展
  
- `inference_moe.py`: 用于模型推理和文本生成的脚本

## 推理脚本使用方法

### 基本用法

```bash
# 使用TransformerMoE模型生成文本
python inference_moe.py --prompt "这是一个测试："

# 使用长序列模型
python inference_moe.py --model_type long_transformer_moe --prompt "请写一篇长文章："

# 使用SimpleMoE模型生成向量
python inference_moe.py --model_type simple_moe
```

### 加载已训练模型

```bash
python inference_moe.py --model_path /path/to/model.pt --vocab_path /path/to/tokenizer.json
```

### 批量处理提示文件

```bash
python inference_moe.py --input_file examples/prompts.txt
```

### 调整生成参数

```bash
python inference_moe.py --prompt "写一个故事：" \
                       --max_new_tokens 200 \
                       --temperature 0.8 \
                       --top_p 0.92 \
                       --top_k 40 \
                       --repetition_penalty 1.05
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model_type` | 模型类型：simple_moe, transformer_moe, long_transformer_moe | transformer_moe |
| `--model_path` | 模型权重路径 | ./checkpoints/model.pt |
| `--vocab_path` | 词表文件路径 | ./data/tokenizer.json |
| `--prompt` | 生成的提示文本 | "" |
| `--input_file` | 输入文件，包含多行文本作为提示 | "" |
| `--max_new_tokens` | 最大生成token数量 | 100 |
| `--temperature` | 采样温度 | 0.7 |
| `--top_p` | Top-p采样阈值 | 0.9 |
| `--top_k` | Top-k采样阈值 | 50 |
| `--repetition_penalty` | 重复惩罚系数 | 1.1 |
| `--seed` | 随机种子 | 42 |

## 模型特点

### SimpleMoE

基本的前馈网络MoE模型，适用于向量到向量的映射任务：

- 输入向量 → 选择专家子网络 → 输出向量
- 每个样本通过门控网络选择最适合的k个专家
- 专家输出根据门控分数加权融合

### TransformerMoE

基于Transformer架构的MoE模型，适用于序列建模和NLP任务：

- 在Transformer编码器架构中引入MoE层
- 使用旋转位置编码（RoPE）处理位置信息
- 每个位置的表示通过MoE层独立处理

### LongContextTransformerMoE

支持超长上下文的TransformerMoE模型：

- 基于YaRN（Yet another RoPE extension）技术扩展上下文长度
- 支持处理16K+的序列长度
- 无需重新训练即可扩展上下文窗口大小

## RoPE位置编码变体

- `RotaryEmbedding`: 基础旋转位置编码实现
- `LinearScalingRotaryEmbedding`: 通过线性缩放位置索引扩展上下文窗口
- `DynamicNTKScalingRotaryEmbedding`: 使用动态NTK缩放方法扩展上下文窗口
- `YarnRotaryEmbedding`: 结合多种技术的高级上下文扩展方法 

### 安装ai_client依赖包

```bash
pip install asyncio websockets vosk requests edge-tts pydub python-dotenv
```
