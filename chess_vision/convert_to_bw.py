"""
彩色棋子图像转黑白模板工具 - 简化版
"""

import cv2
import numpy as np
from pathlib import Path

# ============ 配置区域 ============
INPUT_IMAGE = r"D:\Projects\AIchess\chess_vision\test_template\shuai.png"  # 输入：彩色棋子图像路径
INVERT_COLOR = True        # 是否反转颜色（True=反转为白字黑底，推荐）
# =================================


def convert_to_bw_template(image_path, invert=False):
    """
    将彩色棋子图像转换为黑白模板
    
    Args:
        image_path: 输入图像路径
        invert: 是否反转颜色
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"📂 输入图像: {image_path}")
    print(f"📐 原始尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 1. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 二值化（Otsu自动阈值）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 反转颜色（如果需要）
    if invert:
        binary = cv2.bitwise_not(binary)
        print("✓ 已反转颜色")
    
    # 4. 生成输出文件名
    path = Path(image_path)
    output_path = path.parent / f"{path.stem}_bw{path.suffix}"
    
    # 5. 保存
    cv2.imwrite(str(output_path), binary)
    
    print(f"✅ 已保存: {output_path}")
    print(f"📏 尺寸: {binary.shape[1]} x {binary.shape[0]}")
    
    # 统计信息
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    white_ratio = white_pixels / total_pixels * 100
    
    print(f"📊 白色像素占比: {white_ratio:.1f}%")
    
    if white_ratio < 10:
        print("⚠️  白色像素太少，可能需要反转颜色（设置 INVERT_COLOR = True）")
    elif white_ratio > 90:
        print("⚠️  白色像素太多，可能需要反转颜色（设置 INVERT_COLOR = False）")


if __name__ == "__main__":
    convert_to_bw_template(INPUT_IMAGE, INVERT_COLOR)

