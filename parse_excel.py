"""
Excel文件解析脚本
功能：
- 读取Excel文件
- 提取指定列的数据
- 将数据转换为JSON格式
- 保存到新的JSON文件
"""

import pandas as pd
import json
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_excel_to_json(excel_path, json_path, columns=None, sheet_name=0):
    """
    解析Excel文件并转换为JSON格式
    
    参数:
        excel_path: Excel文件路径
        json_path: 输出JSON文件路径
        columns: 需要提取的列名列表，如果为None则提取所有列
        sheet_name: Excel工作表名称或索引
    """
    try:
        # 读取Excel文件
        logging.info(f"正在读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 如果指定了列，则只提取这些列
        if columns:
            # 检查所有指定的列是否存在
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"以下列在Excel文件中不存在: {missing_columns}")
            df = df[columns]
        
        # 将DataFrame转换为JSON格式
        # orient='records' 表示每行数据作为一个独立的字典
        json_data = df.to_dict(orient='records')
        
        # 创建输出目录（如果不存在）
        output_dir = Path(json_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"成功将数据保存到: {json_path}")
        logging.info(f"共处理 {len(json_data)} 条记录")
        
        return True
        
    except Exception as e:
        logging.error(f"处理Excel文件时出错: {str(e)}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将Excel文件转换为JSON格式')
    parser.add_argument('--input', '-i', required=True,
                      help='输入的Excel文件路径')
    parser.add_argument('--output', '-o', required=True,
                      help='输出的JSON文件路径')
    parser.add_argument('--columns', '-c', nargs='+',
                      help='需要提取的列名列表，用空格分隔')
    parser.add_argument('--sheet', '-s', default=0,
                      help='Excel工作表名称或索引（默认为0）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行转换
    success = parse_excel_to_json(
        excel_path=args.input,
        json_path=args.output,
        columns=args.columns,
        sheet_name=args.sheet
    )
    
    if success:
        logging.info("转换完成！")
    else:
        logging.error("转换失败！")

if __name__ == "__main__":
    main() 