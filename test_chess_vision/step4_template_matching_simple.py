"""
步骤4：模板匹配（多种方法测试）
测试不同的匹配方法，避免拉伸导致的匹配失败
"""

import cv2
import numpy as np
import os

# ============ 用户配置 ============
# 待匹配图像路径（step3 输出的黑白图像）
INPUT_IMAGE = r"D:\Projects\AIchess\test_chess_vision\debug_black_v4_pao\11_final_processed.png"

# 模板图像路径（单个模板）
TEMPLATE_IMAGE = r"D:\Projects\AIchess\PikaShark\templates\black\cannon.png"

# 输出文件夹
OUTPUT_DIR = "step4_output"
# =================================


def preprocess_image_to_gray(image_path):
    """
    预处理图像：转灰度
    
    参数:
        image_path: 图像路径
    
    返回:
        gray_image: 灰度图像
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        return None
    
    # 转灰度图
    if len(image.shape) == 2:
        gray_image = image
    elif image.shape[2] == 4:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    return gray_image


def method1_center_crop_matching(input_img, template_img, output_dir):
    """
    方法1：中心裁剪匹配（推荐）
    将较大的图像裁剪到较小图像的尺寸，保持中心对齐
    不拉伸任何图像，保持原始比例
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
    
    返回:
        match_score: 匹配度
    """
    print("\n" + "="*60)
    print("方法1：中心裁剪匹配")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    
    # 确定目标尺寸（取较小的）
    target_w = min(w1, w2)
    target_h = min(h1, h2)
    
    print(f"目标尺寸: {target_w}x{target_h}")
    
    # 中心裁剪输入图像
    if w1 > target_w or h1 > target_h:
        x1 = (w1 - target_w) // 2
        y1 = (h1 - target_h) // 2
        input_cropped = input_img[y1:y1+target_h, x1:x1+target_w]
        print(f"裁剪输入图像: 从({x1},{y1})裁剪{target_w}x{target_h}")
    else:
        input_cropped = input_img
    
    # 中心裁剪模板图像
    if w2 > target_w or h2 > target_h:
        x2 = (w2 - target_w) // 2
        y2 = (h2 - target_h) // 2
        template_cropped = template_img[y2:y2+target_h, x2:x2+target_w]
        print(f"裁剪模板图像: 从({x2},{y2})裁剪{target_w}x{target_h}")
    else:
        template_cropped = template_img
    
    # 保存处理后的图像
    cv2.imwrite(os.path.join(output_dir, "method1_input_cropped.png"), input_cropped)
    cv2.imwrite(os.path.join(output_dir, "method1_template_cropped.png"), template_cropped)
    
    # 模板匹配
    result = cv2.matchTemplate(input_cropped, template_cropped, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    print(f"\n匹配度: {max_val:.6f}")
    
    return max_val


def method2_sliding_window_matching(input_img, template_img, output_dir):
    """
    方法2：滑动窗口匹配
    模板在输入图像上滑动，找到最佳匹配位置
    适用于模板小于输入图像的情况
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
    
    返回:
        match_score: 匹配度
    """
    print("\n" + "="*60)
    print("方法2：滑动窗口匹配")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    
    # 如果模板大于输入，交换它们
    if w2 > w1 or h2 > h1:
        print("模板大于输入，交换角色")
        input_img, template_img = template_img, input_img
        h1, w1 = input_img.shape
        h2, w2 = template_img.shape
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, "method2_input.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method2_template.png"), template_img)
    
    # 滑动窗口匹配
    result = cv2.matchTemplate(input_img, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    print(f"最佳匹配位置: {max_loc}")
    print(f"匹配度: {max_val:.6f}")
    
    # 可视化匹配位置
    vis_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    top_left = max_loc
    bottom_right = (top_left[0] + w2, top_left[1] + h2)
    cv2.rectangle(vis_img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "method2_match_location.png"), vis_img)
    
    return max_val


def method3_feature_matching(input_img, template_img, output_dir):
    """
    方法3：特征点匹配（ORB）
    提取特征点进行匹配，不受尺度和旋转影响
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
    
    返回:
        match_score: 匹配度（好的匹配点比例）
    """
    print("\n" + "="*60)
    print("方法3：特征点匹配（ORB）")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, "method3_input.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method3_template.png"), template_img)
    
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=500)
    
    # 检测特征点和描述符
    kp1, des1 = orb.detectAndCompute(input_img, None)
    kp2, des2 = orb.detectAndCompute(template_img, None)
    
    print(f"输入图像特征点数: {len(kp1)}")
    print(f"模板图像特征点数: {len(kp2)}")
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print("特征点不足，无法匹配")
        return 0.0
    
    # 使用BFMatcher匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"匹配点数: {len(matches)}")
    
    # 计算匹配度（好的匹配点比例）
    if len(matches) > 0:
        # 取前30%的匹配点
        good_matches = matches[:max(1, len(matches) // 3)]
        match_score = len(good_matches) / max(len(kp1), len(kp2))
        print(f"好的匹配点数: {len(good_matches)}")
        print(f"匹配度: {match_score:.6f}")
    else:
        match_score = 0.0
        print("无匹配点")
    
    # 可视化匹配
    if len(matches) > 0:
        vis_img = cv2.drawMatches(input_img, kp1, template_img, kp2, 
                                   matches[:20], None, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(output_dir, "method3_matches.png"), vis_img)
    
    return match_score


def method4_histogram_correlation(input_img, template_img, output_dir):
    """
    方法4：直方图相关性
    比较两个图像的灰度直方图相似度
    不受尺寸影响，但只能反映整体亮度分布
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
    
    返回:
        match_score: 相关性（0-1）
    """
    print("\n" + "="*60)
    print("方法4：直方图相关性")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, "method4_input.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method4_template.png"), template_img)
    
    # 计算直方图
    hist1 = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([template_img], [0], None, [256], [0, 256])
    
    # 归一化
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # 计算相关性
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    print(f"直方图相关性: {correlation:.6f}")
    
    return correlation


# def method5_ssim_matching(input_img, template_img, output_dir):
#     """
#     方法5：结构相似性（SSIM）
#     需要安装 scikit-image: pip install scikit-image
#     比较两个相同尺寸图像的结构相似度
#     
#     参数:
#         input_img: 待匹配图像（灰度）
#         template_img: 模板图像（灰度）
#         output_dir: 输出目录
#     
#     返回:
#         match_score: SSIM值（0-1）
#     """
#     from skimage.metrics import structural_similarity as ssim
#     
#     print("\n" + "="*60)
#     print("方法5：结构相似性（SSIM）")
#     print("="*60)
#     
#     h1, w1 = input_img.shape
#     h2, w2 = template_img.shape
#     
#     print(f"输入图像尺寸: {w1}x{h1}")
#     print(f"模板图像尺寸: {w2}x{h2}")
#     
#     # 调整到相同尺寸（取较小的）
#     target_w = min(w1, w2)
#     target_h = min(h1, h2)
#     
#     # 中心裁剪
#     x1 = (w1 - target_w) // 2
#     y1 = (h1 - target_h) // 2
#     input_cropped = input_img[y1:y1+target_h, x1:x1+target_w]
#     
#     x2 = (w2 - target_w) // 2
#     y2 = (h2 - target_h) // 2
#     template_cropped = template_img[y2:y2+target_h, x2:x2+target_w]
#     
#     # 保存图像
#     cv2.imwrite(os.path.join(output_dir, "method5_input_cropped.png"), input_cropped)
#     cv2.imwrite(os.path.join(output_dir, "method5_template_cropped.png"), template_cropped)
#     
#     # 计算SSIM
#     score = ssim(input_cropped, template_cropped)
#     
#     print(f"SSIM值: {score:.6f}")
#     
#     return score


def method6_contour_matching(input_img, template_img, output_dir):
    """
    方法6：轮廓匹配
    提取轮廓并比较形状相似度

    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录

    返回:
        match_score: 形状匹配度（值越小越相似）
    """
    print("\n" + "="*60)
    print("方法6：轮廓匹配")
    print("="*60)

    h1, w1 = input_img.shape
    h2, w2 = template_img.shape

    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")

    # 保存图像
    cv2.imwrite(os.path.join(output_dir, "method6_input.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method6_template.png"), template_img)

    # 二值化
    _, binary1 = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours1) == 0 or len(contours2) == 0:
        print("未找到轮廓")
        return 1.0

    # 取最大轮廓
    cnt1 = max(contours1, key=cv2.contourArea)
    cnt2 = max(contours2, key=cv2.contourArea)

    print(f"输入图像轮廓点数: {len(cnt1)}")
    print(f"模板图像轮廓点数: {len(cnt2)}")

    # 形状匹配（Hu矩）
    match_score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0)

    print(f"形状匹配度: {match_score:.6f} (越小越相似)")

    # 可视化轮廓
    vis1 = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    vis2 = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis1, [cnt1], -1, (0, 255, 0), 2)
    cv2.drawContours(vis2, [cnt2], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "method6_contour1.png"), vis1)
    cv2.imwrite(os.path.join(output_dir, "method6_contour2.png"), vis2)

    return match_score


def method7_merged_contour_matching(input_img, template_img, output_dir, noise_threshold=0.02):
    """
    方法7：过滤噪点 + 合并轮廓匹配（推荐）⭐
    提取所有轮廓，过滤小噪点，合并后进行形状匹配
    适合识别汉字等多笔画文字
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
        noise_threshold: 噪点过滤阈值（相对于总面积的比例，默认2%）
    
    返回:
        match_score: 形状匹配度（值越小越相似）
    """
    print("\n" + "="*60)
    print("方法7：过滤噪点 + 合并轮廓匹配")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    print(f"噪点过滤阈值: {noise_threshold*100}% 总面积")
    
    # ========== 步骤0：保存原始图像 ==========
    cv2.imwrite(os.path.join(output_dir, "method7_0_input_original.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method7_0_template_original.png"), template_img)
    
    # ========== 步骤1：二值化 ==========
    print("\n步骤1：二值化")
    _, binary1 = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "method7_1_input_binary.png"), binary1)
    cv2.imwrite(os.path.join(output_dir, "method7_1_template_binary.png"), binary2)
    
    # ========== 步骤2：提取所有轮廓 ==========
    print("\n步骤2：提取所有轮廓")
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours1) == 0 or len(contours2) == 0:
        print("未找到轮廓")
        return 1.0
    
    print(f"输入图像找到 {len(contours1)} 个轮廓")
    print(f"模板图像找到 {len(contours2)} 个轮廓")
    
    # 可视化所有轮廓（不同颜色）
    vis1_all = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    vis2_all = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, cnt in enumerate(contours1):
        color = colors[i % len(colors)]
        cv2.drawContours(vis1_all, [cnt], -1, color, 2)
    for i, cnt in enumerate(contours2):
        color = colors[i % len(colors)]
        cv2.drawContours(vis2_all, [cnt], -1, color, 2)
    cv2.imwrite(os.path.join(output_dir, "method7_2_input_all_contours.png"), vis1_all)
    cv2.imwrite(os.path.join(output_dir, "method7_2_template_all_contours.png"), vis2_all)
    
    # ========== 步骤3：过滤噪点 ==========
    print("\n步骤3：过滤噪点")
    
    # 处理输入图像
    total_area1 = sum(cv2.contourArea(cnt) for cnt in contours1)
    min_area1 = total_area1 * noise_threshold
    
    print(f"\n输入图像:")
    valid_contours1 = []
    filtered_count1 = 0
    for i, cnt in enumerate(contours1):
        area = cv2.contourArea(cnt)
        if area >= min_area1:
            valid_contours1.append(cnt)
            print(f"  轮廓{i+1}: 保留 - 面积={area:.1f} ({area/total_area1*100:.1f}%)")
        else:
            filtered_count1 += 1
            print(f"  轮廓{i+1}: 过滤 - 面积={area:.1f} ({area/total_area1*100:.1f}%)")
    
    print(f"结果: 保留 {len(valid_contours1)} 个，过滤 {filtered_count1} 个")
    
    # 处理模板图像
    total_area2 = sum(cv2.contourArea(cnt) for cnt in contours2)
    min_area2 = total_area2 * noise_threshold
    
    print(f"\n模板图像:")
    valid_contours2 = []
    filtered_count2 = 0
    for i, cnt in enumerate(contours2):
        area = cv2.contourArea(cnt)
        if area >= min_area2:
            valid_contours2.append(cnt)
            print(f"  轮廓{i+1}: 保留 - 面积={area:.1f} ({area/total_area2*100:.1f}%)")
        else:
            filtered_count2 += 1
            print(f"  轮廓{i+1}: 过滤 - 面积={area:.1f} ({area/total_area2*100:.1f}%)")
    
    print(f"结果: 保留 {len(valid_contours2)} 个，过滤 {filtered_count2} 个")
    
    if len(valid_contours1) == 0 or len(valid_contours2) == 0:
        print("过滤后无有效轮廓")
        return 1.0
    
    # 可视化有效轮廓
    vis1_valid = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    vis2_valid = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(valid_contours1):
        color = colors[i % len(colors)]
        cv2.drawContours(vis1_valid, [cnt], -1, color, 2)
    for i, cnt in enumerate(valid_contours2):
        color = colors[i % len(colors)]
        cv2.drawContours(vis2_valid, [cnt], -1, color, 2)
    cv2.imwrite(os.path.join(output_dir, "method7_3_input_valid_contours.png"), vis1_valid)
    cv2.imwrite(os.path.join(output_dir, "method7_3_template_valid_contours.png"), vis2_valid)
    
    # ========== 步骤4：合并轮廓 ==========
    print("\n步骤4：合并轮廓")
    
    # 输入图像：将所有有效轮廓绘制到mask上（填充）
    mask1 = np.zeros_like(binary1)
    cv2.drawContours(mask1, valid_contours1, -1, 255, -1)  # -1表示填充所有轮廓
    cv2.imwrite(os.path.join(output_dir, "method7_4_input_merged_mask.png"), mask1)
    
    # 模板图像：将所有有效轮廓绘制到mask上（填充）
    mask2 = np.zeros_like(binary2)
    cv2.drawContours(mask2, valid_contours2, -1, 255, -1)
    cv2.imwrite(os.path.join(output_dir, "method7_4_template_merged_mask.png"), mask2)
    
    # 从mask中提取合并后的轮廓
    merged_contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(merged_contours1) == 0 or len(merged_contours2) == 0:
        print("合并后无轮廓")
        return 1.0
    
    # 取最大的合并轮廓
    merged_cnt1 = max(merged_contours1, key=cv2.contourArea)
    merged_cnt2 = max(merged_contours2, key=cv2.contourArea)
    
    print(f"输入图像合并后轮廓点数: {len(merged_cnt1)}")
    print(f"模板图像合并后轮廓点数: {len(merged_cnt2)}")
    
    # 可视化合并后的轮廓
    vis1_merged = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    vis2_merged = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis1_merged, [merged_cnt1], -1, (0, 255, 0), 2)
    cv2.drawContours(vis2_merged, [merged_cnt2], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "method7_4_input_merged_contour.png"), vis1_merged)
    cv2.imwrite(os.path.join(output_dir, "method7_4_template_merged_contour.png"), vis2_merged)
    
    # ========== 步骤5：形状匹配 ==========
    print("\n步骤5：形状匹配（Hu矩）")
    match_score = cv2.matchShapes(merged_cnt1, merged_cnt2, cv2.CONTOURS_MATCH_I1, 0)
    
    print(f"匹配度: {match_score:.6f} (越小越相似)")
    
    return match_score


def method0_original_matching(input_img, template_img, output_dir):
    """
    方法0：原始模板匹配方法（从vision.py复制）
    将模板调整到预处理后的格子大小，然后用TM_CCOEFF_NORMED匹配
    
    参数:
        input_img: 待匹配图像（灰度）
        template_img: 模板图像（灰度）
        output_dir: 输出目录
    
    返回:
        match_score: 匹配度
    """
    print("\n" + "="*60)
    print("方法0：原始模板匹配方法（vision.py）")
    print("="*60)
    
    h1, w1 = input_img.shape
    h2, w2 = template_img.shape
    
    print(f"输入图像尺寸: {w1}x{h1}")
    print(f"模板图像尺寸: {w2}x{h2}")
    
    # 调整模板大小到预处理后的格子大小
    cell_size = (w1, h1)
    resized_template = cv2.resize(template_img, cell_size)
    
    print(f"调整模板大小: {w2}x{h2} → {w1}x{h1}")
    
    # 保存调整后的模板
    cv2.imwrite(os.path.join(output_dir, "method0_input.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "method0_template_original.png"), template_img)
    cv2.imwrite(os.path.join(output_dir, "method0_template_resized.png"), resized_template)
    
    # 模板匹配（TM_CCOEFF_NORMED）
    result = cv2.matchTemplate(input_img, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    print(f"\n匹配度: {max_val:.6f}")
    print(f"匹配方法: TM_CCOEFF_NORMED")
    print(f"阈值参考: 0.3 (项目中使用的阈值)")
    
    if max_val >= 0.3:
        print(f"结果: ✅ 匹配成功 (>= 0.3)")
    else:
        print(f"结果: ❌ 匹配失败 (< 0.3)")
    
    return max_val


def main():
    """主函数"""
    print("\n" + "="*60)
    print("步骤4：模板匹配（多种方法测试）")
    print("="*60)
    print("\n说明：")
    print("  - 测试多种不拉伸图像的匹配方法")
    print("  - 避免因比例不同导致的匹配失败")
    print("  - 输出处理后的图像到step4_output文件夹")
    print("\n" + "="*60)
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取图像
    print(f"\n📂 读取图像...")
    input_img = preprocess_image_to_gray(INPUT_IMAGE)
    template_img = preprocess_image_to_gray(TEMPLATE_IMAGE)
    
    if input_img is None or template_img is None:
        print("❌ 无法读取图像")
        return
    
    print(f"✓ 输入图像: {input_img.shape[1]}x{input_img.shape[0]}")
    print(f"✓ 模板图像: {template_img.shape[1]}x{template_img.shape[0]}")
    
    # 保存原始图像
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original_input.png"), input_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original_template.png"), template_img)
    
    # 测试各种方法
    results = {}
    
    # 方法0：原始模板匹配方法（vision.py）⭐ 当前使用
    results['方法0_原始匹配'] = method0_original_matching(input_img, template_img, OUTPUT_DIR)
    
    # 方法1：中心裁剪匹配
    # results['方法1_中心裁剪'] = method1_center_crop_matching(input_img, template_img, OUTPUT_DIR)
    
    # 方法2：滑动窗口匹配
    # results['方法2_滑动窗口'] = method2_sliding_window_matching(input_img, template_img, OUTPUT_DIR)
    
    # 方法3：特征点匹配
    # results['方法3_特征点'] = method3_feature_matching(input_img, template_img, OUTPUT_DIR)
    
    # 方法4：直方图相关性
    # results['方法4_直方图'] = method4_histogram_correlation(input_img, template_img, OUTPUT_DIR)
    
    # 方法6：轮廓匹配（单轮廓）
    # results['方法6_轮廓'] = method6_contour_matching(input_img, template_img, OUTPUT_DIR)
    
    # 方法7：过滤噪点 + 合并轮廓匹配
    # results['方法7_合并轮廓'] = method7_merged_contour_matching(input_img, template_img, OUTPUT_DIR, noise_threshold=0.02)
    
    # 输出结果汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    for method, score in results.items():
        print(f"{method}: {score:.6f}")
    
    print(f"\n✅ 测试完成，结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
