import cv2
import numpy as np

def auto_cut_elements(img_path, output_dir="output"):
    # 读取带透明通道的PNG
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    
    # 预处理参数（可根据需要调整）
    PADDING = 2       # 扩展边缘像素
    MIN_AREA = 100    # 最小元素面积
    GRAY_THRESH = 200 # 二值化阈值
    
    # 分离透明通道
    alpha_channel = img[:, :, 3]
    
    # 创建二值化蒙版
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, GRAY_THRESH, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作（连接相邻区域）
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历并保存元素
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
            
        # 获取边界框（带扩展边缘）
        x,y,w,h = cv2.boundingRect(cnt)
        x = max(0, x - PADDING)
        y = max(0, y - PADDING)
        w = min(w + 2*PADDING, img.shape[1] - x)
        h = min(h + 2*PADDING, img.shape[0] - y)
        
        # 提取元素（保留透明通道）
        element = img[y:y+h, x:x+w]
        
        # 保存为PNG
        cv2.imwrite(f"{output_dir}/element_{i}.png", element)
    
    # 调试显示（可选）
    # cv2.imshow('Processed', eroded)
    # cv2.waitKey(0)

if __name__ == "__main__":
    auto_cut_elements("ui_screenshot.png")