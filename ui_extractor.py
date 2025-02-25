import cv2
import numpy as np
import os

def extract_ui_elements(image_path, output_dir):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 如果图片包含alpha通道，使用它来检测UI元素
    if image.shape[2] == 4:
        # 提取alpha通道
        alpha = image[:, :, 3]
        # 对alpha通道进行二值化
        _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    else:
        # 如果没有alpha通道，转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个找到的轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤掉太小的区域（可以根据需要调整阈值）
        if w < 20 or h < 20:
            continue
            
        # 提取UI元素
        ui_element = image[y:y+h, x:x+w]
        
        # 保存UI元素
        output_path = os.path.join(output_dir, f'ui_element_{i}.png')
        cv2.imwrite(output_path, ui_element)

# 使用示例
if __name__ == "__main__":
    input_image = "input.png"  # 替换为您的输入图片路径
    output_directory = "extracted_elements"  # 输出目录
    extract_ui_elements(input_image, output_directory)