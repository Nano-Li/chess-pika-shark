"""
步骤3：预处理红色棋子
完全按照参考项目的方式处理红色棋子图像
"""

import cv2
import numpy as np
import os

# ============ 配置参数 ============
class PieceProcessConfig:
    # 红色HSV范围（用于提取掩码）
    # 
    # 【优化方法】使用取色器获取棋子的实际RGB值，然后转换为HSV：
    # 1. 在图像软件（如Photoshop、GIMP）中打开棋子图像
    # 2. 使用取色器工具点击棋子的红色区域，获取RGB值
    #    例如：R=220, G=50, B=45
    # 3. 将RGB转换为HSV（可以用在线工具或下面的代码）：
    #    import cv2
    #    import numpy as np
    #    rgb = np.uint8([[[45, 50, 220]]])  # 注意OpenCV是BGR顺序
    #    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    #    print(hsv[0][0])  # 输出：[H, S, V]
    # 4. 根据转换结果调整下面的范围：
    #    - H值 ±5-10° 作为范围
    #    - S值可以适当放宽（比如实际值-20 到 255）
    #    - V值可以适当放宽（比如实际值-30 到 255）
    # 
    # 【当前值】参考项目的默认值，适用于一般红色棋子
    # 
    # 范围1: H=0-10°, S=100-255, V=100-255（红色在HSV色环的低端）
    RED_HSV_LOWER_1 = (0, 100, 100)
    RED_HSV_UPPER_1 = (10, 255, 255)
    
    # 范围2: H=160-179°, S=100-255, V=100-255（红色在HSV色环的高端）
    RED_HSV_LOWER_2 = (160, 100, 100)
    RED_HSV_UPPER_2 = (179, 255, 255)
    
    # 【调试技巧】
    # - 如果红色像素太少（< 5%）：降低S和V的下界（比如改成80）
    # - 如果红色像素太多（> 30%，包含背景）：提高S和V的下界（比如改成120）
    # - 如果提取不完整：扩大H的范围（比如0-15°和155-179°）
    
    # 轮廓筛选参数
    # 【说明】用于筛选出真正的棋子轮廓，排除噪声和背景
    ASPECT_RATIO_MIN = 0.8      # 宽高比最小值（棋子接近正方形）
    ASPECT_RATIO_MAX = 1.2      # 宽高比最大值
    AREA_RATIO_MIN = 0.3        # 面积比最小值（占格子面积的比例）
    
    # 【优化建议】
    # - 如果棋子被裁剪（轮廓太小）：降低 AREA_RATIO_MIN（比如0.2）
    # - 如果检测到背景噪声：提高 AREA_RATIO_MIN（比如0.4）
    # - 如果棋子不是正方形：放宽宽高比范围（比如0.7~1.3）
    
    # 形态学操作参数
    # 【说明】用于填充掩码中的小孔洞，使轮廓更完整
    MORPH_KERNEL_SIZE = 3       # 形态学核大小
    MORPH_ITERATIONS = 3        # 形态学迭代次数
    
    # 【优化建议】
    # - 如果掩码有很多小孔洞：增加 MORPH_ITERATIONS（比如5）
    # - 如果掩码边缘被过度平滑：减少 MORPH_ITERATIONS（比如1）

# ============ 用户配置 ============
# 输入图像路径（格子图像）
INPUT_IMAGE = "./template_cells/cells/cell_04-05.png"  # 修改这里指定要处理的格子图像

# 输出目录
# OUTPUT_DIR = "step3_output"
OUTPUT_DIR = "template_red_output"

# 是否保存中间结果
SAVE_INTERMEDIATE = True
# =================================


def extract_red_piece_mask(cell_image):
    """
    提取红色棋子的掩码
    
    参数:
        cell_image: BGR格式的格子图像
    
    返回:
        mask: 二值掩码（红色区域=255，其他=0）
    """
    print("\n步骤1：提取红色掩码")
    print("-"*60)
    
    # 1. 转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    print(f"   转换为HSV颜色空间")
    
    # 2. 创建红色掩码（范围1）
    mask1 = cv2.inRange(hsv_image, 
                       PieceProcessConfig.RED_HSV_LOWER_1, 
                       PieceProcessConfig.RED_HSV_UPPER_1)
    print(f"   范围1: H={PieceProcessConfig.RED_HSV_LOWER_1[0]}-{PieceProcessConfig.RED_HSV_UPPER_1[0]}°, "
          f"S={PieceProcessConfig.RED_HSV_LOWER_1[1]}-{PieceProcessConfig.RED_HSV_UPPER_1[1]}, "
          f"V={PieceProcessConfig.RED_HSV_LOWER_1[2]}-{PieceProcessConfig.RED_HSV_UPPER_1[2]}")
    
    # 3. 创建红色掩码（范围2）
    mask2 = cv2.inRange(hsv_image, 
                       PieceProcessConfig.RED_HSV_LOWER_2, 
                       PieceProcessConfig.RED_HSV_UPPER_2)
    print(f"   范围2: H={PieceProcessConfig.RED_HSV_LOWER_2[0]}-{PieceProcessConfig.RED_HSV_UPPER_2[0]}°, "
          f"S={PieceProcessConfig.RED_HSV_LOWER_2[1]}-{PieceProcessConfig.RED_HSV_UPPER_2[1]}, "
          f"V={PieceProcessConfig.RED_HSV_LOWER_2[2]}-{PieceProcessConfig.RED_HSV_UPPER_2[2]}")
    
    # 4. 合并两个掩码
    mask_red = cv2.bitwise_or(mask1, mask2)
    
    # 统计红色像素
    red_pixels = cv2.countNonZero(mask_red)
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    red_ratio = red_pixels / total_pixels
    
    print(f"   红色像素: {red_pixels}/{total_pixels} ({red_ratio*100:.2f}%)")
    
    return mask_red


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
    print("\n步骤2：提取最大轮廓区域")
    print("-"*60)
    
    # 1. 形态学闭运算（填充小孔洞）
    kernel = np.ones((PieceProcessConfig.MORPH_KERNEL_SIZE, 
                     PieceProcessConfig.MORPH_KERNEL_SIZE), np.uint8)
    morphed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, 
                                    iterations=PieceProcessConfig.MORPH_ITERATIONS)
    print(f"   形态学闭运算: kernel={PieceProcessConfig.MORPH_KERNEL_SIZE}x{PieceProcessConfig.MORPH_KERNEL_SIZE}, "
          f"iterations={PieceProcessConfig.MORPH_ITERATIONS}")
    
    # 2. 查找轮廓
    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   检测到 {len(contours)} 个轮廓")
    
    if len(contours) == 0:
        print("   ⚠️  未检测到轮廓，返回原始掩码")
        return original_mask, None
    
    # 3. 筛选有效轮廓
    img_area = mask_image.shape[0] * mask_image.shape[1]
    valid_contours = []
    
    print(f"   筛选条件:")
    print(f"     - 宽高比: {PieceProcessConfig.ASPECT_RATIO_MIN} ~ {PieceProcessConfig.ASPECT_RATIO_MAX}")
    print(f"     - 面积比: >= {PieceProcessConfig.AREA_RATIO_MIN}")
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        contour_area = w * h
        area_ratio = contour_area / img_area
        
        print(f"   轮廓{i+1}: 位置=({x},{y}), 尺寸={w}x{h}, "
              f"宽高比={aspect_ratio:.2f}, 面积比={area_ratio:.2f}")
        
        # 条件1：接近正方形
        if PieceProcessConfig.ASPECT_RATIO_MIN <= aspect_ratio <= PieceProcessConfig.ASPECT_RATIO_MAX:
            # 条件2：占格子面积 >= 50%
            if area_ratio >= PieceProcessConfig.AREA_RATIO_MIN:
                valid_contours.append({
                    'contour': contour,
                    'area': contour_area,
                    'rect': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area_ratio': area_ratio
                })
                print(f"     ✓ 有效轮廓")
            else:
                print(f"     ✗ 面积比不足")
        else:
            print(f"     ✗ 宽高比不符")
    
    # 4. 选择最大的轮廓
    if len(valid_contours) == 0:
        print("   ⚠️  没有有效轮廓，返回原始掩码")
        return original_mask, None
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    print(f"\n   选择最大轮廓:")
    print(f"     位置: ({x}, {y})")
    print(f"     尺寸: {w}x{h}")
    print(f"     宽高比: {best_contour['aspect_ratio']:.2f}")
    print(f"     面积比: {best_contour['area_ratio']:.2f}")
    
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


def process_red_piece(cell_image, save_intermediate=True, output_dir="output"):
    """
    处理红色棋子图像
    
    参数:
        cell_image: BGR格式的格子图像
        save_intermediate: 是否保存中间结果
        output_dir: 输出目录
    
    返回:
        processed_image: 处理后的图像
        info: 处理信息字典
    """
    print("\n" + "="*60)
    print("开始处理红色棋子")
    print("="*60)
    print(f"输入图像尺寸: {cell_image.shape[1]}x{cell_image.shape[0]}")
    
    # 创建输出目录
    if save_intermediate:
        os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：提取红色掩码
    mask_red = extract_red_piece_mask(cell_image)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, "01_red_mask.png"), mask_red)
        print(f"   💾 已保存: 01_red_mask.png")
    
    # 步骤2：提取最大轮廓区域
    processed_image, contour_info = extract_largest_contour_region(mask_red, mask_red)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, "02_morphed_mask.png"), 
                   cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, 
                                   np.ones((3, 3), np.uint8), iterations=3))
        print(f"   💾 已保存: 02_morphed_mask.png")
        
        # 在原图上绘制轮廓
        if contour_info:
            debug_image = cell_image.copy()
            x, y, w, h = contour_info['x'], contour_info['y'], contour_info['width'], contour_info['height']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "03_contour_debug.png"), debug_image)
            print(f"   💾 已保存: 03_contour_debug.png")
    
    # 步骤3：保存最终结果
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, "04_final_processed.png"), processed_image)
        print(f"   💾 已保存: 04_final_processed.png")
    
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
    print("步骤3：预处理红色棋子")
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
    
    # 处理红色棋子
    processed_image, info = process_red_piece(cell_image, SAVE_INTERMEDIATE, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ 全部完成！")
    print("="*60)
    print(f"📁 输出文件:")
    if SAVE_INTERMEDIATE:
        print(f"   {OUTPUT_DIR}/00_original.png         - 原始格子图像")
        print(f"   {OUTPUT_DIR}/01_red_mask.png         - 红色掩码")
        print(f"   {OUTPUT_DIR}/02_morphed_mask.png     - 形态学处理后的掩码")
        print(f"   {OUTPUT_DIR}/03_contour_debug.png    - 轮廓调试图像")
        print(f"   {OUTPUT_DIR}/04_final_processed.png  - 最终处理结果")
    
    # 显示图像（可选）
    print("\n按任意键关闭窗口...")
    cv2.imshow("Original", cell_image)
    cv2.imshow("Processed", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

