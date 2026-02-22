"""
步骤3B：预处理黑色棋子（调试版）
测试不同的Canny阈值，找到最佳参数
"""

import cv2
import numpy as np
import os

# ============ 用户配置 ============
# 输入图像路径（格子图像）
INPUT_IMAGE = "./step1_output/cell_04-05.png"

# 输出目录
OUTPUT_DIR = "step3b_debug_output"

# Canny阈值测试范围
CANNY_THRESHOLDS = [
    (100, 200),  # 低阈值
    (200, 400),  # 中低阈值
    (300, 500),  # 中阈值
    (400, 600),  # 高阈值（参考项目使用）
    (500, 700),  # 超高阈值
]
# =================================


def test_canny_thresholds(cell_image, output_dir):
    """
    测试不同的Canny阈值
    """
    print("\n" + "="*60)
    print("测试不同的Canny阈值")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 转HSV + 直方图均衡化
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    
    # 保存均衡化后的图像
    cv2.imwrite(os.path.join(output_dir, "00_original.png"), cell_image)
    cv2.imwrite(os.path.join(output_dir, "01_hsv_equalized.png"), 
               cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR))
    
    print(f"✓ 已保存: 00_original.png")
    print(f"✓ 已保存: 01_hsv_equalized.png")
    
    # 2. 测试不同的Canny阈值
    for i, (thresh1, thresh2) in enumerate(CANNY_THRESHOLDS):
        print(f"\n测试 {i+1}/{len(CANNY_THRESHOLDS)}: Canny阈值=({thresh1}, {thresh2})")
        print("-"*60)
        
        # Canny边缘检测
        edges = cv2.Canny(hsv_equalized, thresh1, thresh2)
        
        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"   检测到 {len(contours)} 个轮廓")
        
        # 保存边缘图像
        edge_filename = f"02_edges_{thresh1}_{thresh2}.png"
        cv2.imwrite(os.path.join(output_dir, edge_filename), edges)
        
        # 保存形态学处理后的边缘
        morphed_filename = f"03_morphed_{thresh1}_{thresh2}.png"
        cv2.imwrite(os.path.join(output_dir, morphed_filename), morphed_edges)
        
        # 在原图上绘制所有轮廓
        contour_image = cell_image.copy()
        
        if len(contours) > 0:
            # 按面积排序
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 绘制前3个最大的轮廓
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # 绿、红、蓝
            for j, contour in enumerate(sorted_contours[:3]):
                area = cv2.contourArea(contour)
                color = colors[j] if j < len(colors) else (128, 128, 128)
                cv2.drawContours(contour_image, [contour], 0, color, 2)
                
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                print(f"   轮廓{j+1}: 面积={area:.0f}, 位置=({x},{y}), 尺寸={w}x{h}")
        
        # 保存轮廓图像
        contour_filename = f"04_contours_{thresh1}_{thresh2}.png"
        cv2.imwrite(os.path.join(output_dir, contour_filename), contour_image)
        
        print(f"   ✓ 已保存: {edge_filename}")
        print(f"   ✓ 已保存: {morphed_filename}")
        print(f"   ✓ 已保存: {contour_filename}")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"\n💡 检查要点:")
    print(f"   1. 查看 02_edges_*.png - 哪个阈值能检测到外圈轮廓？")
    print(f"   2. 查看 04_contours_*.png - 哪个阈值的轮廓最完整？")
    print(f"   3. 绿色=最大轮廓, 红色=第2大, 蓝色=第3大")


def main():
    """主函数"""
    print("="*60)
    print("步骤3B：预处理黑色棋子（调试版）")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 输入文件不存在: {INPUT_IMAGE}")
        return
    
    print(f"📂 输入图像: {INPUT_IMAGE}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    
    # 读取图像
    cell_image = cv2.imread(INPUT_IMAGE)
    if cell_image is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    
    print(f"📐 图像尺寸: {cell_image.shape[1]}x{cell_image.shape[0]}")
    
    # 测试不同的Canny阈值
    test_canny_thresholds(cell_image, OUTPUT_DIR)


if __name__ == "__main__":
    main()

