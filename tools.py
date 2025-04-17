#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from pathlib import Path


def parquet_to_jsonl(input_path, output_path=None):
    """
    将Parquet格式的数据文件转换为JSONL格式
    
    参数:
        input_path (str): Parquet文件路径
        output_path (str, optional): JSONL输出文件路径，如果为None，则使用与input_path相同的文件名但扩展名为.jsonl
    
    返回:
        str: 输出文件的路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件 {input_path} 不存在")
    
    # 如果没有指定输出路径，则使用相同的文件名但扩展名为.jsonl
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.with_suffix('.jsonl'))
    
    # 读取parquet文件
    print(f"正在读取Parquet文件: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 将数据转换为JSONL格式并写入文件
    print(f"正在转换为JSONL格式并写入: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 将每一行转换为JSON字符串并写入文件
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    
    print(f"转换完成! 共处理 {len(df)} 条记录")
    return output_path


def batch_parquet_to_jsonl(input_dir, output_dir=None, pattern="*.parquet"):
    """
    批量将目录下的所有Parquet文件转换为JSONL格式
    
    参数:
        input_dir (str): 包含Parquet文件的目录
        output_dir (str, optional): JSONL输出文件的目录，如果为None，则与input_dir相同
        pattern (str, optional): 用于匹配Parquet文件的模式，默认为"*.parquet"
    
    返回:
        list: 所有输出文件的路径列表
    """
    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"目录 {input_dir} 不存在")
    
    # 如果没有指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = input_dir
    else:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有匹配的Parquet文件
    input_dir_path = Path(input_dir)
    parquet_files = list(input_dir_path.glob(pattern))
    
    if not parquet_files:
        print(f"在目录 {input_dir} 中未找到匹配模式 {pattern} 的Parquet文件")
        return []
    
    output_files = []
    for parquet_file in parquet_files:
        # 构建输出文件路径
        output_file = Path(output_dir) / parquet_file.with_suffix('.jsonl').name
        
        # 转换文件
        output_path = parquet_to_jsonl(str(parquet_file), str(output_file))
        output_files.append(output_path)
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将Parquet文件转换为JSONL格式")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径或目录")
    parser.add_argument("--output", "-o", help="输出文件路径或目录(可选)")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理目录下的所有Parquet文件")
    parser.add_argument("--pattern", "-p", default="*.parquet", help="在批处理模式下用于匹配文件的模式(默认: *.parquet)")
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            output_files = batch_parquet_to_jsonl(args.input, args.output, args.pattern)
            if output_files:
                print(f"批量转换完成! 共生成 {len(output_files)} 个JSONL文件:")
                for output_file in output_files:
                    print(f"  - {output_file}")
        else:
            output_path = parquet_to_jsonl(args.input, args.output)
            print(f"单文件转换完成! 输出文件: {output_path}")
    except Exception as e:
        print(f"错误: {e}") 