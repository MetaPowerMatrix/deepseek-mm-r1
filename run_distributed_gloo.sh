 #!/bin/bash
# 使用Gloo后端的分布式训练启动脚本
# 当NCCL出现问题时可以使用此脚本作为备选

# 确保脚本在错误时退出
set -e

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 如果没有检测到GPU，则退出
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "错误: 未检测到可用的GPU。分布式训练需要至少一个GPU。"
    exit 1
fi

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

# 设置PyTorch和CUDA环境变量
export PYTHONFAULTHANDLER=1   # 启用Python错误处理器，打印更详细的错误信息
export CUDA_LAUNCH_BLOCKING=1 # 使CUDA启动同步，有助于调试
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 设置PyTorch分布式调试级别

# 强制使用Gloo后端
export TORCH_DISTRIBUTED_BACKEND=gloo  # 强制使用gloo作为分布式后端

# 创建保存目录
mkdir -p $SAVE_DIR
mkdir -p $(dirname $DATA_PATH)

echo "启动分布式训练 (Gloo后端): 使用 $NUM_GPUS 个 GPU"
echo "模型类型: $MODEL_TYPE"
echo "批次大小: $BATCH_SIZE (每GPU) x $NUM_GPUS (GPU数) = $(($BATCH_SIZE*$NUM_GPUS)) (总批次大小)"
echo "保存目录: $SAVE_DIR"

# 使用torchrun启动分布式训练 (使用GLOO后端)
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --log_dir=$SAVE_DIR/logs \
    --backend=gloo \
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