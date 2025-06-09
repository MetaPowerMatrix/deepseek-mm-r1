#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIF到JPG转换器
将指定目录下的所有GIF图片转换为JPG格式
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse


def convert_gif_to_jpg(gif_path, output_dir=None, quality=95):
    """
    将单个GIF文件转换为JPG格式
    
    Args:
        gif_path (str): GIF文件路径
        output_dir (str): 输出目录，如果为None则保存在原目录
        quality (int): JPG质量 (1-100)
    
    Returns:
        str: 转换后的JPG文件路径，如果失败返回None
    """
    try:
        # 打开GIF文件
        with Image.open(gif_path) as img:
            # 如果是动态GIF，取第一帧
            if hasattr(img, 'is_animated') and img.is_animated:
                img.seek(0)  # 选择第一帧
            
            # 转换为RGB模式（JPG不支持透明度）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 生成输出文件路径
            gif_path = Path(gif_path)
            if output_dir:
                output_path = Path(output_dir) / f"{gif_path.stem}.jpg"
            else:
                output_path = gif_path.parent / f"{gif_path.stem}.jpg"
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            print(f"✓ 转换成功: {gif_path.name} -> {output_path.name}")
            return str(output_path)
            
    except Exception as e:
        print(f"✗ 转换失败: {gif_path.name} - {str(e)}")
        return None


def find_gif_files(directory):
    """
    在指定目录及其子目录中查找所有GIF文件
    
    Args:
        directory (str): 搜索目录
    
    Returns:
        list: GIF文件路径列表
    """
    gif_files = []
    directory = Path(directory)
    
    # 支持的GIF文件扩展名
    gif_extensions = {'.gif', '.GIF'}
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in gif_extensions:
            gif_files.append(file_path)
    
    return gif_files


def convert_directory(input_dir, output_dir=None, quality=95, recursive=True):
    """
    转换目录下的所有GIF文件
    
    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录，如果为None则保存在原目录
        quality (int): JPG质量 (1-100)
        recursive (bool): 是否递归搜索子目录
    
    Returns:
        tuple: (成功数量, 失败数量)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 目录 '{input_dir}' 不存在")
        return 0, 0
    
    if not input_path.is_dir():
        print(f"错误: '{input_dir}' 不是一个目录")
        return 0, 0
    
    # 查找GIF文件
    if recursive:
        gif_files = find_gif_files(input_dir)
    else:
        gif_files = [f for f in input_path.iterdir() 
                    if f.is_file() and f.suffix.lower() == '.gif']
    
    if not gif_files:
        print(f"在目录 '{input_dir}' 中没有找到GIF文件")
        return 0, 0
    
    print(f"找到 {len(gif_files)} 个GIF文件")
    print("开始转换...")
    
    success_count = 0
    failure_count = 0
    
    for gif_file in gif_files:
        result = convert_gif_to_jpg(gif_file, output_dir, quality)
        if result:
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {failure_count} 个文件")
    
    return success_count, failure_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将目录下的所有GIF图片转换为JPG格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python gif_to_jpg_converter.py                     # 转换当前目录下的GIF文件
  python gif_to_jpg_converter.py /path/to/images     # 转换指定目录下的GIF文件
  python gif_to_jpg_converter.py -o /path/to/output  # 指定输出目录
  python gif_to_jpg_converter.py -q 85               # 设置JPG质量为85
  python gif_to_jpg_converter.py --no-recursive      # 不递归搜索子目录
        """
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        default='.',
        help='输入目录路径 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='输出目录路径 (默认: 与原文件相同目录)'
    )
    
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=95,
        choices=range(1, 101),
        metavar='1-100',
        help='JPG质量 1-100 (默认: 95)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录'
    )
    
    args = parser.parse_args()
    
    # 检查Pillow库是否可用
    try:
        from PIL import Image
    except ImportError:
        print("错误: 需要安装Pillow库")
        print("请运行: pip install Pillow")
        sys.exit(1)
    
    print("GIF到JPG转换器")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    else:
        print("输出目录: 与原文件相同目录")
    print(f"JPG质量: {args.quality}")
    print(f"递归搜索: {'否' if args.no_recursive else '是'}")
    print("=" * 50)
    
    # 执行转换
    success, failure = convert_directory(
        args.input_dir,
        args.output_dir,
        args.quality,
        not args.no_recursive
    )
    
    # 返回适当的退出码
    if failure > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()