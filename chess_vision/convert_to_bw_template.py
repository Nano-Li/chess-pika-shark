"""
彩色棋子图像转黑白模板工具
模仿 xiangqi-analysis 项目的模板风格
"""

import cv2
import numpy as np
from pathlib import Path

# ============ 配置区域 ============
INPUT_IMAGE = r"D:\Projects\AIchess\chess_vision\test_template\jiang.png"  # 输入：彩色棋子图像
# =================================


def convert_to_bw_template(image_path):
    """
    将彩色棋子图像转换为黑白模板
    
    处理步骤：
    1. 读取彩色图像
    2. 转灰度
    3. 对比度拉满（二值化）
    4. 反转颜色（棋子白色，背景黑色）
    5. 保存
    
    Args:
        image_path: 输入图像路径
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
    print("✓ 已转灰度")
    
    # 2. 对比度拉满（二值化）
    # 方法1：简单阈值
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 方法2：自适应阈值（更智能）
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)
    
    # 方法3：Otsu自动阈值（推荐）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("✓ 已二值化")
    
    # 3. 反转颜色（可选，取决于你的图像）
    # 如果棋子是黑色，背景是白色，需要反转
    # binary = cv2.bitwise_not(binary)
    
    # 检查：棋子应该是白色（255），背景应该是黑色（0）
    # 如果反了，取消下面这行的注释
    # binary = cv2.bitwise_not(binary)
    
    # 4. 生成输出文件名
    path = Path(image_path)
    output_path = path.parent / f"{path.stem}_bw{path.suffix}"
    
    # 5. 保存
    cv2.imwrite(str(output_path), binary)
    
    print(f"✅ 已保存黑白模板: {output_path}")
    print(f"📏 输出尺寸: {binary.shape[1]} x {binary.shape[0]}")
    print()
    
    # 显示统计信息
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    total_pixels = binary.shape[0] * binary.shape[1]
    
    print(f"📊 像素统计:")
    print(f"   白色像素: {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    print(f"   黑色像素: {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
    
    if white_pixels < black_pixels * 0.1:
        print()
        print("⚠️  警告：白色像素太少，可能需要反转颜色")
        print("   取消代码中 'binary = cv2.bitwise_not(binary)' 的注释")


def batch_convert(input_dir="."):
    """
    批量转换目录下的所有PNG图像
    
    Args:
        input_dir: 输入目录
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png"))
    
    # 排除已经是_bw结尾的文件
    image_files = [f for f in image_files if not f.stem.endswith('_bw')]
    
    if not image_files:
        print("❌ 未找到PNG图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print()
    
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 处理: {img_file.name}")
        print("-" * 60)
        convert_to_bw_template(str(img_file))
        print()
    
    print("=" * 60)
    print("🎉 批量转换完成！")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("彩色棋子图像 → 黑白模板转换工具")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        # 命令行参数：单个文件
        convert_to_bw_template(sys.argv[1])
    else:
        # 交互模式
        print("选择模式:")
        print("  1) 转换单个文件")
        print("  2) 批量转换当前目录所有PNG")
        print()
        
        choice = input("请选择 (1/2): ").strip()
        
        if choice == '1':
            # 单个文件
            image_path = input("\n请输入图像路径: ").strip().strip('"')
            convert_to_bw_template(image_path)
        elif choice == '2':
            # 批量转换
            input_dir = input("\n请输入目录路径 (默认: 当前目录): ").strip().strip('"')
            if not input_dir:
                input_dir = "."
            batch_convert(input_dir)
        else:
            print("无效选择")

