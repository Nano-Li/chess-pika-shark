"""
图像边缘裁剪工具
裁剪图像的上/下/左/右边缘指定像素
"""

import cv2
from pathlib import Path

# ============ 配置区域 ============
IMAGE_PATH = r"D:\Projects\AIchess\tiantian00.png"  # 修改为你的图像路径

# 裁剪像素数（从边缘向内裁剪）
# CROP_TOP = 2      # 上边裁剪像素
# CROP_BOTTOM = 0   # 下边裁剪像素
# CROP_LEFT = 5     # 左边裁剪像素
# CROP_RIGHT = 0    # 右边裁剪像素

CROP_TOP = 4      # 上边裁剪像素
CROP_BOTTOM = 2   # 下边裁剪像素
CROP_LEFT = 6     # 左边裁剪像素
CROP_RIGHT = 2    # 右边裁剪像素
# =================================

def crop_image(image_path, top, bottom, left, right):
    """裁剪图像边缘"""
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"📐 原始尺寸: {w} x {h}")
    
    # 计算裁剪后的区域
    y1 = top
    y2 = h - bottom
    x1 = left
    x2 = w - right
    
    # 检查有效性
    if y2 <= y1 or x2 <= x1:
        print(f"❌ 裁剪参数无效！")
        print(f"   上:{top} 下:{bottom} 左:{left} 右:{right}")
        return
    
    # 裁剪
    cropped = image[y1:y2, x1:x2]
    
    new_h, new_w = cropped.shape[:2]
    print(f"📏 裁剪后尺寸: {new_w} x {new_h}")
    print(f"✂️  裁剪: 上{top}px, 下{bottom}px, 左{left}px, 右{right}px")
    
    # 生成输出文件名
    path = Path(image_path)
    output_path = path.parent / f"{path.stem}_cropped{path.suffix}"
    
    # 保存
    cv2.imwrite(str(output_path), cropped)
    
    print(f"✅ 已保存到: {output_path.absolute()}")


if __name__ == "__main__":
    crop_image(IMAGE_PATH, CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT)

