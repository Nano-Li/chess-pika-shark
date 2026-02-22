"""
棋盘模板分割工具 - 简化版
直接分割棋盘为90个格子，保存到output文件夹
"""

import cv2
import os
from pathlib import Path

# ============ 配置区域 ============
BOARD_IMAGE_PATH = r"D:\Projects\AIchess\tiantian_wx.png"  # 修改为你的棋盘图像路径
# =================================

def split_board(board_image_path):
    """分割棋盘为90个格子"""
    
    # 读取图像
    image = cv2.imread(board_image_path)
    if image is None:
        print(f"❌ 无法读取图像: {board_image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"📐 棋盘尺寸: {w} x {h}")
    
    # 计算格子尺寸
    rows, cols = 10, 9
    cell_height = h // rows
    cell_width = w // cols
    
    print(f"📏 格子尺寸: {cell_width} x {cell_height}")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"💾 保存位置: {output_dir.absolute()}")
    print()
    
    # 分割并保存
    for i in range(rows):
        for j in range(cols):
            # 计算格子位置
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            
            # 裁剪格子
            cell = image[y1:y2, x1:x2]
            
            # 生成文件名：行-列.png（从1开始）
            filename = f"{i+1}-{j+1}.png"
            filepath = output_dir / filename
            
            # 保存
            cv2.imwrite(str(filepath), cell)
    
    print(f"✅ 完成！已保存90个格子到: {output_dir.absolute()}")


if __name__ == "__main__":
    split_board(BOARD_IMAGE_PATH)

