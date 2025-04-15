#!/bin/bash

# 设置环境变量
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi -L | wc -l)

# 启动分布式训练
python train_distributed.py

# 打印完成信息
echo "Training completed on $NUM_GPUS GPUs."