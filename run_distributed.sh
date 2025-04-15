#!/bin/bash
# 分布式训练启动脚本

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 设置训练参数
MODEL_TYPE="transformer_moe"  # 模型类型: simple_moe, transformer_moe, long_transformer_moe
BATCH_SIZE=8                  # 每个GPU的批次大小
NUM_EPOCHS=5                  # 训练轮次
LEARNING_RATE=5e-5            # 学习率
MAX_SEQ_LENGTH=512            # 最大序列长度
GRADIENT_ACCUMULATION=4       # 梯度累积步数
FP16="--fp16"                 # 混合精度训练 (移除此参数禁用)
DATA_PATH="./data/distill_r1_110k_sft.jsonl"
VOCAB_PATH="./data/tokenizer.json"
SAVE_DIR="./checkpoints_moe_distributed"

# 模型结构参数
D_MODEL=768                   # 模型隐藏层维度
NUM_HEADS=12                  # 注意力头数
NUM_LAYERS=6                  # Transformer层数
D_FF=3072                     # 前馈网络维度
NUM_EXPERTS=8                 # 专家数量 
K=2                           # 每次激活的专家数量

# 设置分布式训练环境变量
export NCCL_DEBUG=INFO        # 设置NCCL调试级别 (可选: INFO, WARNING, ERROR)
export OMP_NUM_THREADS=1      # 控制每个进程使用的OpenMP线程数

# 使用torchrun启动分布式训练 (替代已弃用的torch.distributed.launch)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_moe.py \
    --model_type=$MODEL_TYPE \
    --data_path=$DATA_PATH \
    --vocab_path=$VOCAB_PATH \
    --save_dir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --gradient_accumulation=$GRADIENT_ACCUMULATION \
    --d_model=$D_MODEL \
    --num_heads=$NUM_HEADS \
    --num_layers=$NUM_LAYERS \
    --d_ff=$D_FF \
    --num_experts=$NUM_EXPERTS \
    --k=$K \
    --distributed \
    $FP16

echo "分布式训练完成！"