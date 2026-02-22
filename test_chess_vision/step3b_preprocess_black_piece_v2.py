"""
步骤3B：预处理黑色棋子（复杂方法 v2）
完全复刻参考项目的黑色棋子处理方法
"""

import cv2
import numpy as np
import os

# ============ 配置参数 ============
class BlackPieceProcessConfig:
    # 黑色HSV范围
    BLACK_HSV_LOWER = (0, 0, 0)
    BLACK_HSV_UPPER = (180, 255, 80)
    
    # Canny边缘检测参数（高阈值）
    CANNY_THRESHOLD1 = 400
    CANNY_THRESHOLD2 = 600
    
    # 轮廓缩放因子
    CONTOUR_SCALE_FACTOR = 1.00
    
    # 形态学操作参数
    MORPH_KERNEL_SIZE = 3
    MORPH_ITERATIONS = 3
    
    # 轮廓筛选参数
    ASPECT_RATIO_MIN = 0.3
    ASPECT_RATIO_MAX = 1.8
    AREA_RATIO_MIN = 0.1

# ============ 用户配置 ============
# 输入图像路径（格子图像）
INPUT_IMAGE = r"D:\Projects\AIchess\test_chess_vision\step1_output\cell_02-04.png"  # 修改这里指定要处理的格子图像
# INPUT_IMAGE = "black_zu.png"  # 修改这里指定要处理的格子图像

# 输出目录
# OUTPUT_DIR = "step3b_v2_old_method"
OUTPUT_DIR = "debug_black_v2_jiang"

# 是否保存中间结果
SAVE_INTERMEDIATE = True
# =================================


def extract_black_piece_mask_complex(cell_image, save_debug=False, output_dir="output"):
    """
    提取黑色棋子的掩码（复杂方法）
    完全复刻参考项目的方法
    
    参数:
        cell_image: BGR格式的格子图像
        save_debug: 是否保存调试图像
        output_dir: 输出目录
    
    返回:
        mask_black: 二值掩码（黑色区域=255，其他=0）
    """
    print("\n步骤1：提取黑色掩码（复杂方法）")
    print("-"*60)
    
    # 1. 转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    print(f"   1.1 转换为HSV颜色空间")
    
    # 2. 直方图均衡化（增强对比度）
    h, s, v = cv2.split(hsv_image)
    v_equalized = cv2.equalizeHist(v)
    hsv_image = cv2.merge([h, s, v_equalized])
    print(f"   1.2 V通道直方图均衡化（增强对比度）")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "01_hsv_equalized.png"), 
                   cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))
        print(f"       💾 已保存: 01_hsv_equalized.png")
    
    # 3. Canny边缘检测（高阈值）
    edges = cv2.Canny(hsv_image, 
                     BlackPieceProcessConfig.CANNY_THRESHOLD1,
                     BlackPieceProcessConfig.CANNY_THRESHOLD2)
    print(f"   1.3 Canny边缘检测: 阈值=({BlackPieceProcessConfig.CANNY_THRESHOLD1}, {BlackPieceProcessConfig.CANNY_THRESHOLD2})")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "02_edges.png"), edges)
        print(f"       💾 已保存: 02_edges.png")
    
    # 4. 形态学闭运算（连接边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    print(f"   1.4 形态学闭运算（连接边缘）")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "03_morphed_edges.png"), morphed_edges)
        print(f"       💾 已保存: 03_morphed_edges.png")
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   1.5 检测到 {len(contours)} 个轮廓")
    
    if len(contours) == 0:
        print("   ⚠️  未检测到轮廓，返回空掩码")
        return np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    
    # 6. 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(max_contour)
    print(f"   1.6 最大轮廓面积: {max_area:.0f}")
    
    # 7. 计算轮廓中心
    M = cv2.moments(max_contour)
    if M['m00'] == 0:
        print("   ⚠️  轮廓矩为0，返回空掩码")
        return np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    
    center_x = M['m10'] / M['m00']
    center_y = M['m01'] / M['m00']
    print(f"   1.7 轮廓中心: ({center_x:.1f}, {center_y:.1f})")
    
    # 8. 轮廓缩小0.98倍（避免包含背景）
    scaled_contour = []
    for point in max_contour:
        x, y = point[0]
        new_x = center_x + (x - center_x) * BlackPieceProcessConfig.CONTOUR_SCALE_FACTOR
        new_y = center_y + (y - center_y) * BlackPieceProcessConfig.CONTOUR_SCALE_FACTOR
        scaled_contour.append([[int(new_x), int(new_y)]])
    
    scaled_contour = np.array(scaled_contour, dtype=np.int32)
    print(f"   1.8 轮廓缩放: 因子={BlackPieceProcessConfig.CONTOUR_SCALE_FACTOR}")
    
    if save_debug:
        debug_contour = cell_image.copy()
        cv2.drawContours(debug_contour, [max_contour], 0, (0, 255, 0), 2)
        cv2.drawContours(debug_contour, [scaled_contour], 0, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "04_contours.png"), debug_contour)
        print(f"       💾 已保存: 04_contours.png (绿色=原始轮廓, 红色=缩放后)")
    
    # 9. 创建黑色掩码
    # 转回原始HSV（不用均衡化的）
    hsv_original = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_original,
                            BlackPieceProcessConfig.BLACK_HSV_LOWER,
                            BlackPieceProcessConfig.BLACK_HSV_UPPER)
    print(f"   1.9 创建黑色掩码: V=0-80")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "05_black_mask.png"), black_mask)
        print(f"       💾 已保存: 05_black_mask.png")
    
    # 10. 创建轮廓掩码
    largest_contour_mask = np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    cv2.drawContours(largest_contour_mask, [scaled_contour], 0, 255, cv2.FILLED)
    print(f"   1.10 创建轮廓掩码")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "06_contour_mask.png"), largest_contour_mask)
        print(f"       💾 已保存: 06_contour_mask.png")
    
    # 11. 两个掩码相交（黑色掩码 AND 轮廓掩码）
    final_mask = cv2.bitwise_and(black_mask, largest_contour_mask)
    print(f"   1.11 掩码相交（黑色 AND 轮廓）")
    
    # 统计黑色像素
    black_pixels = cv2.countNonZero(final_mask)
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    black_ratio = black_pixels / total_pixels
    print(f"   1.12 最终黑色像素: {black_pixels}/{total_pixels} ({black_ratio*100:.2f}%)")
    
    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "07_final_mask.png"), final_mask)
        print(f"       💾 已保存: 07_final_mask.png")
    
    return final_mask


def extract_largest_contour_region(mask_image, original_mask):
    """
    提取最大轮廓区域
    
    参数:
        mask_image: 输入掩码图像
        original_mask: 原始掩码（用于裁剪）
    
    返回:
        cropped_image: 裁剪后的图像（只包含棋子）
        contour_info: 轮廓信息字典
    """
    print("\n步骤2：提取最大轮廓区域并裁剪")
    print("-"*60)
    
    # 1. 形态学闭运算（填充小孔洞）
    kernel = np.ones((BlackPieceProcessConfig.MORPH_KERNEL_SIZE, 
                     BlackPieceProcessConfig.MORPH_KERNEL_SIZE), np.uint8)
    morphed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, 
                                    iterations=BlackPieceProcessConfig.MORPH_ITERATIONS)
    print(f"   2.1 形态学闭运算: kernel={BlackPieceProcessConfig.MORPH_KERNEL_SIZE}x{BlackPieceProcessConfig.MORPH_KERNEL_SIZE}, "
          f"iterations={BlackPieceProcessConfig.MORPH_ITERATIONS}")
    
    # 2. 查找轮廓
    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   2.2 检测到 {len(contours)} 个轮廓")
    
    if len(contours) == 0:
        print("   ⚠️  未检测到轮廓，返回原始掩码")
        return original_mask, None
    
    # 3. 筛选有效轮廓
    img_area = mask_image.shape[0] * mask_image.shape[1]
    valid_contours = []
    
    print(f"   2.3 筛选条件:")
    print(f"       - 宽高比: {BlackPieceProcessConfig.ASPECT_RATIO_MIN} ~ {BlackPieceProcessConfig.ASPECT_RATIO_MAX}")
    print(f"       - 面积比: >= {BlackPieceProcessConfig.AREA_RATIO_MIN}")
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        contour_area = w * h
        area_ratio = contour_area / img_area
        
        print(f"       轮廓{i+1}: 位置=({x},{y}), 尺寸={w}x{h}, "
              f"宽高比={aspect_ratio:.2f}, 面积比={area_ratio:.2f}")
        
        # 条件1：接近正方形
        if BlackPieceProcessConfig.ASPECT_RATIO_MIN <= aspect_ratio <= BlackPieceProcessConfig.ASPECT_RATIO_MAX:
            # 条件2：占格子面积 >= 30%
            if area_ratio >= BlackPieceProcessConfig.AREA_RATIO_MIN:
                valid_contours.append({
                    'contour': contour,
                    'area': contour_area,
                    'rect': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area_ratio': area_ratio
                })
                print(f"         ✓ 有效轮廓")
            else:
                print(f"         ✗ 面积比不足")
        else:
            print(f"         ✗ 宽高比不符")
    
    # 4. 选择最大的轮廓
    if len(valid_contours) == 0:
        print("   ⚠️  没有有效轮廓，返回原始掩码")
        return original_mask, None
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    print(f"\n   2.4 选择最大轮廓:")
    print(f"       位置: ({x}, {y})")
    print(f"       尺寸: {w}x{h}")
    print(f"       宽高比: {best_contour['aspect_ratio']:.2f}")
    print(f"       面积比: {best_contour['area_ratio']:.2f}")
    
    # 5. 裁剪出轮廓区域
    cropped_image = original_mask[y:y+h, x:x+w]
    
    contour_info = {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'aspect_ratio': best_contour['aspect_ratio'],
        'area_ratio': best_contour['area_ratio']
    }
    
    return cropped_image, contour_info


def process_black_piece_complex(cell_image, save_intermediate=True, output_dir="output"):
    """
    处理黑色棋子图像（复杂方法）
    完全复刻参考项目的方法
    
    参数:
        cell_image: BGR格式的格子图像
        save_intermediate: 是否保存中间结果
        output_dir: 输出目录
    
    返回:
        processed_image: 处理后的图像
        info: 处理信息字典
    """
    print("\n" + "="*60)
    print("开始处理黑色棋子（复杂方法 - 完全复刻参考项目）")
    print("="*60)
    print(f"输入图像尺寸: {cell_image.shape[1]}x{cell_image.shape[0]}")
    
    # 创建输出目录
    if save_intermediate:
        os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：提取黑色掩码（复杂方法）
    mask_black = extract_black_piece_mask_complex(cell_image, save_intermediate, output_dir)
    
    # 步骤2：提取最大轮廓区域
    processed_image, contour_info = extract_largest_contour_region(mask_black, mask_black)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, "08_morphed_for_crop.png"), 
                   cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, 
                                   np.ones((3, 3), np.uint8), iterations=3))
        print(f"   💾 已保存: 08_morphed_for_crop.png")
        
        # 在原图上绘制轮廓
        if contour_info:
            debug_image = cell_image.copy()
            x, y, w, h = contour_info['x'], contour_info['y'], contour_info['width'], contour_info['height']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "09_crop_debug.png"), debug_image)
            print(f"   💾 已保存: 09_crop_debug.png")
    
    # 步骤3：保存最终结果
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, "10_final_processed.png"), processed_image)
        print(f"   💾 已保存: 10_final_processed.png")
    
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"输出图像尺寸: {processed_image.shape[1]}x{processed_image.shape[0]}")
    
    if contour_info:
        print(f"轮廓信息:")
        print(f"  位置: ({contour_info['x']}, {contour_info['y']})")
        print(f"  尺寸: {contour_info['width']}x{contour_info['height']}")
        print(f"  宽高比: {contour_info['aspect_ratio']:.3f}")
        print(f"  面积比: {contour_info['area_ratio']:.3f}")
    
    info = {
        'input_size': (cell_image.shape[1], cell_image.shape[0]),
        'output_size': (processed_image.shape[1], processed_image.shape[0]),
        'contour_info': contour_info
    }
    
    return processed_image, info


def main():
    """主函数"""
    print("="*60)
    print("步骤3B：预处理黑色棋子（复杂方法 v2）")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 输入文件不存在: {INPUT_IMAGE}")
        print(f"   请先运行 step1_detect_chessboard_v2.py 生成格子图像")
        return
    
    print(f"📂 输入图像: {INPUT_IMAGE}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"💾 保存中间结果: {SAVE_INTERMEDIATE}")
    
    # 读取图像
    cell_image = cv2.imread(INPUT_IMAGE)
    if cell_image is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    
    # 保存原始图像到输出目录
    if SAVE_INTERMEDIATE:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "00_original.png"), cell_image)
        print(f"💾 已保存原始图像: 00_original.png")
    
    # 处理黑色棋子
    processed_image, info = process_black_piece_complex(cell_image, SAVE_INTERMEDIATE, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ 全部完成！")
    print("="*60)
    print(f"📁 输出文件:")
    if SAVE_INTERMEDIATE:
        print(f"   {OUTPUT_DIR}/00_original.png           - 原始格子图像")
        print(f"   {OUTPUT_DIR}/01_hsv_equalized.png      - 直方图均衡化后")
        print(f"   {OUTPUT_DIR}/02_edges.png              - Canny边缘检测")
        print(f"   {OUTPUT_DIR}/03_morphed_edges.png      - 形态学闭运算")
        print(f"   {OUTPUT_DIR}/04_contours.png           - 轮廓对比（绿=原始，红=缩放）")
        print(f"   {OUTPUT_DIR}/05_black_mask.png         - 黑色掩码")
        print(f"   {OUTPUT_DIR}/06_contour_mask.png       - 轮廓掩码")
        print(f"   {OUTPUT_DIR}/07_final_mask.png         - 最终掩码（相交）")
        print(f"   {OUTPUT_DIR}/08_morphed_for_crop.png   - 形态学处理（用于裁剪）")
        print(f"   {OUTPUT_DIR}/09_crop_debug.png         - 裁剪调试")
        print(f"   {OUTPUT_DIR}/10_final_processed.png    - 最终处理结果")
    
    # 显示图像（可选）
    print("\n按任意键关闭窗口...")
    cv2.imshow("Original", cell_image)
    cv2.imshow("Processed", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

