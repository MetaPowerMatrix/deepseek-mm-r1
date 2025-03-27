#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网页内容解析脚本
功能：
- 读取visited_urls.txt文件中的URL列表
- 访问每个URL并解析页面内容
- 模式1: 提取class为img-wrap的div中的img标签，下载图片
- 模式2: 提取class为mod tab-item mod-attributes的div中的表格数据
- 将图片保存到prod_imgs目录，或将表格数据保存为JSON文件
"""

import os
import re
import time
import json
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# User-Agent头，模拟浏览器防止被封
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

def extract_order_id(url):
    """
    从URL中提取order_entry_id参数值
    
    参数:
        url: 网页URL
    返回:
        str: order_entry_id值或None
    """
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # 获取order_entry_id参数值
        if 'order_entry_id' in query_params:
            return query_params['order_entry_id'][0]
    except Exception as e:
        logging.error(f"URL解析错误: {url}, 错误: {str(e)}")
    
    return None

def get_page_content(url, max_retries=3):
    """
    获取网页内容
    
    参数:
        url: 网页URL
        max_retries: 最大重试次数
    返回:
        str: 网页内容或None
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                logging.warning(f"获取页面失败: {url}, 状态码: {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"请求错误 (尝试 {attempt+1}/{max_retries}): {url}, 错误: {str(e)}")
            time.sleep(2)  # 等待后重试
    
    logging.error(f"获取页面失败，达到最大重试次数: {url}")
    return None

def extract_image_url(html_content):
    """
    从HTML内容中提取class为img-wrap的div中的img标签的src属性
    
    参数:
        html_content: 网页HTML内容
    返回:
        str: 图片URL或None
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找class为img-wrap的div
        img_wrap_div = soup.find('div', class_='img-wrap')
        
        if img_wrap_div:
            # 在div中查找img标签
            img_tag = img_wrap_div.find('img')
            
            if img_tag and img_tag.has_attr('src'):
                # 获取图片URL
                img_url = img_tag['src']
                
                # 确保URL是完整的
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                
                return img_url
            else:
                logging.warning("在img-wrap中未找到img标签或src属性")
        else:
            logging.warning("未找到class为img-wrap的div")
    
    except Exception as e:
        logging.error(f"解析HTML时出错: {str(e)}")
    
    return None

def extract_table_data(html_content, prod_id):
    """
    从HTML内容中提取class为mod tab-item mod-attributes的div中的表格数据
    
    参数:
        html_content: 网页HTML内容
        prod_id: 产品ID（order_entry_id）
    返回:
        list: 表格数据列表，每项为包含prod_id、label和value的字典
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找class为mod tab-item mod-attributes的div
        attributes_div = soup.find('div', class_='mod tab-item mod-attributes')
        
        if not attributes_div:
            logging.warning("未找到class为'mod tab-item mod-attributes'的div")
            return []
        
        # 在div中查找class为content的div
        content_div = attributes_div.find('div', class_='content')
        
        if not content_div:
            logging.warning("未找到class为'content'的div")
            return []
        
        # 在content_div中查找table标签
        table = content_div.find('table')
        
        if not table:
            logging.warning("未找到table标签")
            return []
        
        # 提取表格中的所有单元格数据
        table_data = []
        for td in table.find_all('td'):
            cell_text = td.get_text(strip=True)
            
            # 处理包含冒号的单元格
            if ':' in cell_text:
                label, value = cell_text.split(':', 1)
                table_data.append({
                    'prod_id': prod_id,
                    'label': label.strip(),
                    'value': value.strip()
                })
        
        return table_data
    
    except Exception as e:
        logging.error(f"解析表格数据时出错: {str(e)}")
        return []

def download_image(img_url, file_path, max_retries=3):
    """
    下载图片并保存到指定路径
    
    参数:
        img_url: 图片URL
        file_path: 保存路径
        max_retries: 最大重试次数
    返回:
        bool: 是否成功下载
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(img_url, headers=HEADERS, timeout=10, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"图片已保存: {file_path}")
                return True
            else:
                logging.warning(f"下载图片失败: {img_url}, 状态码: {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"下载错误 (尝试 {attempt+1}/{max_retries}): {img_url}, 错误: {str(e)}")
            time.sleep(2)  # 等待后重试
    
    logging.error(f"下载图片失败，达到最大重试次数: {img_url}")
    return False

def process_urls_for_images(urls, output_dir='prod_imgs'):
    """
    处理URL列表，下载图片
    
    参数:
        urls: URL列表
        output_dir: 图片保存目录
    返回:
        tuple: (总URL数, 成功下载数, 失败数)
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 统计变量
    total_urls = len(urls)
    successful_downloads = 0
    failed_urls = 0
    
    for i, url in enumerate(urls, 1):
        logging.info(f"处理URL [{i}/{total_urls}]: {url}")
        
        # 从URL中提取order_entry_id
        order_id = extract_order_id(url)
        
        if not order_id:
            logging.warning(f"无法从URL中提取order_entry_id: {url}")
            failed_urls += 1
            continue
        
        # 获取网页内容
        html_content = get_page_content(url)
        
        if not html_content:
            failed_urls += 1
            continue
        
        # 提取图片URL
        img_url = extract_image_url(html_content)
        
        if not img_url:
            logging.warning(f"未找到图片URL: {url}")
            failed_urls += 1
            continue
        
        # 构建图片保存路径
        img_ext = os.path.splitext(img_url.split('?')[0])[1] or '.jpg'  # 获取扩展名，默认为.jpg
        file_path = os.path.join(output_dir, f"{order_id}{img_ext}")
        
        # 下载图片
        if download_image(img_url, file_path):
            successful_downloads += 1
        else:
            failed_urls += 1
        
        # 每处理10个URL后暂停一下，避免请求过于频繁
        if i % 10 == 0:
            logging.info(f"已处理 {i}/{total_urls} 个URL，暂停2秒...")
            time.sleep(2)
    
    return total_urls, successful_downloads, failed_urls

def process_urls_for_table_data(urls, output_file='product_data.json'):
    """
    处理URL列表，提取表格数据并保存为JSON文件
    
    参数:
        urls: URL列表
        output_file: 输出JSON文件路径
    返回:
        tuple: (总URL数, 成功提取数, 失败数)
    """
    # 统计变量
    total_urls = len(urls)
    successful_extractions = 0
    failed_urls = 0
    all_table_data = []
    
    for i, url in enumerate(urls, 1):
        logging.info(f"处理URL [{i}/{total_urls}]: {url}")
        
        # 从URL中提取order_entry_id
        order_id = extract_order_id(url)
        
        if not order_id:
            logging.warning(f"无法从URL中提取order_entry_id: {url}")
            failed_urls += 1
            continue
        
        # 获取网页内容
        html_content = get_page_content(url)
        
        if not html_content:
            failed_urls += 1
            continue
        
        # 提取表格数据
        table_data = extract_table_data(html_content, order_id)
        
        if table_data:
            all_table_data.extend(table_data)
            successful_extractions += 1
            logging.info(f"成功提取 {len(table_data)} 条表格数据: {url}")
        else:
            logging.warning(f"未提取到表格数据: {url}")
            failed_urls += 1
        
        # 每处理10个URL后暂停一下，避免请求过于频繁
        if i % 10 == 0:
            logging.info(f"已处理 {i}/{total_urls} 个URL，暂停2秒...")
            time.sleep(2)
    
    # 保存JSON数据到文件
    if all_table_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_table_data, f, ensure_ascii=False, indent=2)
            logging.info(f"表格数据已保存到: {output_file}")
        except Exception as e:
            logging.error(f"保存JSON文件失败: {str(e)}")
    else:
        logging.warning("没有提取到任何表格数据")
    
    return total_urls, successful_extractions, failed_urls

def process_urls_from_file(file_path, mode='image', output_dir='prod_imgs', output_file='product_data.json'):
    """
    处理URL文件中的每一行URL
    
    参数:
        file_path: URL文件路径
        mode: 处理模式，'image'或'table'
        output_dir: 图片保存目录
        output_file: 表格数据输出文件
    """
    try:
        # 读取URL列表
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if not urls:
            logging.warning(f"文件 {file_path} 中没有找到有效的URL")
            return
        
        logging.info(f"从文件中读取了 {len(urls)} 个URL")
        
        # 根据模式处理URL
        if mode == 'image':
            total_urls, successful_ops, failed_urls = process_urls_for_images(urls, output_dir)
            logging.info("="*50)
            logging.info(f"图片下载完成: 总URL数: {total_urls}, 成功下载: {successful_ops}, 失败: {failed_urls}")
            logging.info(f"图片已保存到目录: {os.path.abspath(output_dir)}")
        elif mode == 'table':
            total_urls, successful_ops, failed_urls = process_urls_for_table_data(urls, output_file)
            logging.info("="*50)
            logging.info(f"表格数据提取完成: 总URL数: {total_urls}, 成功提取: {successful_ops}, 失败: {failed_urls}")
            logging.info(f"数据已保存到文件: {os.path.abspath(output_file)}")
        else:
            logging.error(f"不支持的模式: {mode}")
    
    except Exception as e:
        logging.error(f"处理URL文件时出错: {str(e)}")
        import traceback
        logging.error(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从URL列表文件中下载图片或提取表格数据')
    parser.add_argument('--input-file', '-i', default='visited_urls.txt',
                        help='输入URL列表文件，默认为visited_urls.txt')
    parser.add_argument('--mode', '-m', choices=['image', 'table'], default='image',
                        help='处理模式: image(下载图片) 或 table(提取表格数据)，默认为image')
    parser.add_argument('--output-dir', '-d', default='prod_imgs',
                        help='图片保存目录，默认为prod_imgs，仅在image模式下有效')
    parser.add_argument('--output-file', '-f', default='product_data.json',
                        help='表格数据输出文件，默认为product_data.json，仅在table模式下有效')
    
    args = parser.parse_args()
    
    process_urls_from_file(
        file_path=args.input_file,
        mode=args.mode,
        output_dir=args.output_dir,
        output_file=args.output_file
    ) 