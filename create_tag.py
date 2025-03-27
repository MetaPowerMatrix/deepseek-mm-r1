"""
标签图片生成脚本
功能：
- 生成指定尺寸的白色背景PNG图片
- 默认尺寸为96mm x 47mm
- 支持自定义尺寸和输出路径
- 从JSON文件读取数据并在图片上绘制文字
- 添加二维码图片
- 支持标题自动折行
- 支持批量处理JSON数据生成多个标签
"""

from PIL import Image, ImageDraw, ImageFont
import argparse
import logging
import json
from pathlib import Path
import textwrap
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def wrap_text(text, font, max_width):
    """
    将文本按最大宽度折行
    
    参数:
        text: 要折行的文本
        font: 字体对象
        max_width: 最大宽度（像素）
    返回:
        list: 折行后的文本行列表
    """
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_width = font.getlength(word + ' ')
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_white_tag(width_mm=96, height_mm=47, dpi=300, output_path='tag.png', title=None, price=None, size=None, order_id=None, qrcode_path='qrcode.jpg'):
    """
    创建白色背景的标签图片
    
    参数:
        width_mm: 宽度（毫米）
        height_mm: 高度（毫米）
        dpi: 图片分辨率（每英寸像素数）
        output_path: 输出文件路径
        title: 要显示的商品标题
        price: 要显示的价格（不再显示）
        size: 要显示的尺寸
        order_id: 要显示的订单号
        qrcode_path: 二维码图片路径
    """
    try:
        # 将毫米转换为像素
        # 1英寸 = 25.4毫米
        width_px = int(width_mm * dpi / 25.4)
        height_px = int(height_mm * dpi / 25.4)
        
        # 创建白色背景图片
        image = Image.new('RGB', (width_px, height_px), 'white')
        draw = ImageDraw.Draw(image)
        
        # 设置字体（使用系统默认字体）
        try:
            # 尝试使用系统中文字体
            title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)  # 标题字体
            info_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 16)   # 信息字体
            note_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 18, index=1)   # 底部文字字体，加粗
        except:
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 22)
                info_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 16)
                note_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 18)  # 使用Medium字重作为粗体
            except:
                try:
                    title_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 22)
                    info_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 16)
                    note_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Bold.ttc", 18)  # 尝试使用Bold字重
                except:
                    try:
                        title_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 22)
                        info_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
                        note_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 18)  # Windows黑体已经是粗体
                    except:
                        try:
                            title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
                            info_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
                            note_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 18)  # 使用微软雅黑粗体
                        except:
                            title_font = info_font = note_font = ImageFont.load_default()
        
        # 如果有标题，绘制到图片上
        if title is not None:
            # 设置固定的二维码尺寸
            qrcode_size = 333
            
            # 计算文字区域的最大宽度（总宽度减去二维码宽度和边距）
            text_area_width = width_px - qrcode_size - 40  # 40是左右边距
            
            # 划分文本区域：上部分用于标题，中部分用于尺寸和订单，底部用于鼓励文字
            text_area_height = height_px
            
            # 处理标题折行
            title_lines = wrap_text(title, title_font, text_area_width - 20)  # 左右留10px边距
            
            # 计算标题的高度
            title_line_height = title_font.size + 5  # 5是行间距
            total_title_height = len(title_lines) * title_line_height
            
            # 计算信息区域的高度
            info_line_height = info_font.size + 8  # 增加行间距
            info_area_height = 2 * info_line_height  # 尺寸和订单号两行
            
            # 计算底部注释行数和高度
            note_text = "愿这份隐秘的温柔，为你带来恰到好处的快乐与安心。"
            note_lines = wrap_text(note_text, note_font, text_area_width - 20)
            note_line_height = note_font.size + 3
            note_area_height = len(note_lines) * note_line_height
            
            # 计算所有内容的总高度
            content_total_height = total_title_height + info_area_height
            
            # 计算内容区域的垂直起始位置（垂直居中）
            content_start_y = (height_px - content_total_height) / 2
            
            # 标题起始位置
            title_start_y = content_start_y
            
            # 绘制标题（每行居中）
            for i, line in enumerate(title_lines):
                line_bbox = draw.textbbox((0, 0), line, font=title_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = 20 + (text_area_width - 20 - line_width) / 2
                draw.text((line_x, title_start_y + i * title_line_height), line, font=title_font, fill='black')
            
            # 信息区域起始位置
            info_start_y = title_start_y + total_title_height + 15  # 标题下方15px
            
            # 绘制尺寸信息（居中对齐）
            size_text = f"尺寸: {size}"
            size_bbox = draw.textbbox((0, 0), size_text, font=info_font)
            size_width = size_bbox[2] - size_bbox[0]
            size_x = 20 + (text_area_width - 20 - size_width) / 2
            draw.text((size_x, info_start_y), size_text, font=info_font, fill='black')
            
            # 绘制订单号信息（居中对齐）
            order_text = f"货号: {order_id}"
            order_bbox = draw.textbbox((0, 0), order_text, font=info_font)
            order_width = order_bbox[2] - order_bbox[0]
            order_x = 20 + (text_area_width - 20 - order_width) / 2
            draw.text((order_x, info_start_y + info_line_height), order_text, font=info_font, fill='black')
            
            # 绘制底部鼓励文字（居中），上移至内容区域下方
            note_start_y = height_px - note_area_height - 60  # 更靠上一些，距底部60px
            
            for i, line in enumerate(note_lines):
                line_bbox = draw.textbbox((0, 0), line, font=note_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = 20 + (text_area_width - 20 - line_width) / 2
                draw.text((line_x, note_start_y + i * note_line_height), line, font=note_font, fill='black')
            
            # 添加二维码图片
            try:
                # 检查二维码文件是否存在
                if not Path(qrcode_path).exists():
                    logging.error(f"二维码图片文件不存在: {qrcode_path}")
                    return False
                
                # 打开二维码图片
                qrcode = Image.open(qrcode_path)
                logging.info(f"成功加载二维码图片: {qrcode_path}")
                logging.info(f"二维码原始尺寸: {qrcode.size}")
                
                # 调整二维码大小
                qrcode = qrcode.resize((qrcode_size, qrcode_size))
                logging.info(f"二维码调整后尺寸: {qrcode.size}")
                
                # 计算二维码位置（右侧居中）
                qrcode_x = width_px - qrcode_size - 20  # 20是右边距
                qrcode_y = (height_px - qrcode_size) // 2  # 使用整除
                logging.info(f"二维码位置: x={qrcode_x}, y={qrcode_y}")
                
                # 粘贴二维码
                image.paste(qrcode, (int(qrcode_x), int(qrcode_y)))
                logging.info("二维码已成功添加到图片中")
            except Exception as e:
                logging.error(f"添加二维码图片时出错: {str(e)}")
                logging.error(f"错误类型: {type(e).__name__}")
                import traceback
                logging.error(f"错误堆栈: {traceback.format_exc()}")
        
        # 创建输出目录（如果不存在）
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图片
        image.save(output_path, 'PNG', dpi=(dpi, dpi))
        
        logging.info(f"成功创建标签图片: {output_path}")
        logging.info(f"图片尺寸: {width_px}x{height_px}像素")
        logging.info(f"分辨率: {dpi} DPI")
        
        return True
        
    except Exception as e:
        logging.error(f"创建标签图片时出错: {str(e)}")
        import traceback
        logging.error(f"错误堆栈: {traceback.format_exc()}")
        return False

def format_order_id(order_id):
    """将订单号转换为普通数字字符串，避免科学计数法表示"""
    if order_id is None:
        return '保密发货'
    
    # 如果是数字类型
    if isinstance(order_id, (int, float)):
        # 对于整数值，直接转为整数字符串
        if isinstance(order_id, int) or float(order_id).is_integer():
            return str(int(order_id))
        # 对于浮点数，使用字符串格式化避免科学计数法
        else:
            return f"{order_id:.15f}".rstrip('0').rstrip('.')
    
    # 如果已经是字符串但可能是科学计数法形式
    elif isinstance(order_id, str):
        # 尝试转换为数字再格式化
        try:
            num = float(order_id)
            if num.is_integer():
                return str(int(num))
            else:
                return f"{num:.15f}".rstrip('0').rstrip('.')
        except:
            # 如果不是数字格式的字符串，直接返回
            return order_id
    
    # 其他类型转为字符串
    return str(order_id)

def read_product_data(json_path):
    """
    从JSON文件读取所有商品数据
    
    参数:
        json_path: JSON文件路径
    返回:
        list: 包含(title, price, size, order_id)元组的列表
    """
    products = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data and isinstance(data, list):
                for product in data:
                    title = product.get('title', '谁用谁知道')
                    # 确保title是字符串类型
                    if not isinstance(title, str):
                        title = str(title) if title is not None else '谁用谁知道'
                    
                    price = product.get('price')
                    # 确保price是数字类型
                    if price is None:
                        price = 0.0
                    elif not isinstance(price, (int, float)):
                        try:
                            price = float(price)
                        except:
                            price = 0.0
                    
                    # 获取size和order_id字段
                    size = product.get('size', '均码')
                    # 确保size是字符串类型
                    if not isinstance(size, str):
                        size = str(size) if size is not None else '均码'
                    
                    # 获取订单号并格式化，避免科学计数法
                    order_id_raw = product.get('order_id', '保密发货')
                    order_id = format_order_id(order_id_raw)
                    
                    products.append((title, price, size, order_id))
                        
        logging.info(f"从JSON文件中读取了 {len(products)} 条商品记录")
    except Exception as e:
        logging.error(f"读取JSON文件时出错: {str(e)}")
    
    return products

def batch_create_tags(width_mm, height_mm, dpi, output_dir, json_path, qrcode_path):
    """
    批量创建标签图片
    
    参数:
        width_mm: 宽度（毫米）
        height_mm: 高度（毫米）
        dpi: 图片分辨率（每英寸像素数）
        output_dir: 输出目录
        json_path: JSON数据文件路径
        qrcode_path: 二维码图片路径
    """
    # 读取所有商品数据
    products = read_product_data(json_path)
    
    if not products:
        logging.warning("没有找到有效的商品数据")
        return False
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 批量创建标签图片
    success_count = 0
    for i, (title, price, size, order_id) in enumerate(products):
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"tag{i+1}.png")
        
        # 创建标签图片
        logging.info(f"开始处理第 {i+1} 个商品: {title}")
        if create_white_tag(
            width_mm=width_mm,
            height_mm=height_mm,
            dpi=dpi,
            output_path=output_path,
            title=title,
            price=price,
            size=size,
            order_id=order_id,
            qrcode_path=qrcode_path
        ):
            success_count += 1
    
    logging.info(f"批量处理完成: 成功 {success_count} / 总计 {len(products)}")
    return success_count > 0

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成白色背景的标签图片')
    parser.add_argument('--width', '-w', type=float, default=96,
                      help='标签宽度（毫米），默认为96mm')
    parser.add_argument('--height', '-H', type=float, default=47,
                      help='标签高度（毫米），默认为47mm')
    parser.add_argument('--dpi', '-d', type=int, default=300,
                      help='图片分辨率（DPI），默认为300')
    parser.add_argument('--output-dir', '-o', default='tags',
                      help='输出目录路径，默认为tags')
    parser.add_argument('--json', '-j', default='products.json',
                      help='JSON数据文件路径，默认为products.json')
    parser.add_argument('--qrcode', '-q', default='qrcode.jpg',
                      help='二维码图片路径，默认为qrcode.jpg')
    parser.add_argument('--single', '-s', type=int,
                      help='仅处理指定索引的单个商品（索引从1开始）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.single:
        # 处理单个商品
        products = read_product_data(args.json)
        if not products or args.single > len(products) or args.single < 1:
            logging.error(f"指定的商品索引无效: {args.single}")
            return
        
        # 获取指定索引的商品数据（索引从1开始）
        title, price, size, order_id = products[args.single - 1]
        output_path = os.path.join(args.output_dir, f"tag{args.single}.png")
        
        # 创建标签图片
        success = create_white_tag(
            width_mm=args.width,
            height_mm=args.height,
            dpi=args.dpi,
            output_path=output_path,
            title=title,
            price=price,
            size=size,
            order_id=order_id,
            qrcode_path=args.qrcode
        )
        
        if success:
            logging.info(f"标签图片生成完成: {output_path}")
        else:
            logging.error(f"标签图片生成失败: {output_path}")
    else:
        # 批量处理所有商品
        success = batch_create_tags(
            width_mm=args.width,
            height_mm=args.height,
            dpi=args.dpi,
            output_dir=args.output_dir,
            json_path=args.json,
            qrcode_path=args.qrcode
        )
        
        if success:
            logging.info("所有标签图片生成完成！")
        else:
            logging.error("标签图片生成失败！")

if __name__ == "__main__":
    main()