"""
步骤4：模板匹配
完全复刻参考项目的模板匹配方法
"""

import cv2
import numpy as np
import os
import glob

# ============ 配置参数 ============
class TemplateMatchingConfig:
    # 模板匹配方法
    # cv2.TM_CCOEFF_NORMED: 归一化相关系数匹配
    # 返回值范围：-1 到 1，值越大越相似
    MATCHING_METHOD = cv2.TM_CCOEFF_NORMED
    
    # 匹配阈值
    # 只有匹配度 > 阈值才认为匹配成功
    MATCHING_THRESHOLD = 0.3
    
    # 棋子类型映射
    PIECE_TYPE_MAP = {
        'king': '将/帅',
        'guard': '士',
        'bishop': '象/相',
        'knight': '马',
        'rook': '车',
        'cannon': '炮',
        'pawn': '兵/卒'
    }

# ============ 用户配置 ============
# 模板目录（包含所有模板图像）
TEMPLATE_DIR = "templates"  # 模板文件夹路径

# 待匹配图像路径
INPUT_IMAGE = "step3_output/04_final_processed.png"  # 预处理后的棋子图像

# 棋子颜色（用于筛选模板）
PIECE_COLOR = "red"  # "red" 或 "black"

# 输出目录
OUTPUT_DIR = "step4_output"

# 是否保存调试图像
SAVE_DEBUG = True
# =================================


def preprocess_template(template_path):
    """
    预处理模板图像（完全复刻参考项目）
    
    参数:
        template_path: 模板图像路径
    
    返回:
        gray_template: 灰度模板图像
        template_name: 模板名称
    """
    # 1. 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    
    if template is None:
        return None, None
    
    # 2. 转灰度图（唯一的预处理）
    if len(template.shape) == 2:
        # 已经是灰度图
        gray_template = template
    elif template.shape[2] == 4:
        # RGBA → 灰度
        gray_template = cv2.cvtColor(template, cv2.COLOR_RGBA2GRAY)
    elif template.shape[2] == 3:
        # RGB → 灰度
        gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    else:
        gray_template = template
    
    # 3. 提取模板名称
    template_name = os.path.basename(template_path).split('.')[0]
    
    return gray_template, template_name


def load_all_templates(template_dir):
    """
    加载所有模板
    
    参数:
        template_dir: 模板目录
    
    返回:
        templates: 字典 {模板名称: 灰度模板图像}
    """
    print("\n" + "="*60)
    print("加载模板")
    print("="*60)
    
    if not os.path.exists(template_dir):
        print(f"❌ 模板目录不存在: {template_dir}")
        return {}
    
    # 查找所有图像文件
    template_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        template_files.extend(glob.glob(os.path.join(template_dir, ext)))
    
    if len(template_files) == 0:
        print(f"❌ 模板目录中没有图像文件: {template_dir}")
        return {}
    
    print(f"📂 模板目录: {template_dir}")
    print(f"📊 找到 {len(template_files)} 个模板文件")
    
    templates = {}
    
    for template_path in template_files:
        gray_template, template_name = preprocess_template(template_path)
        
        if gray_template is not None:
            templates[template_name] = gray_template
            print(f"   ✓ {template_name}: {gray_template.shape[1]}x{gray_template.shape[0]}")
        else:
            print(f"   ✗ 无法加载: {template_path}")
    
    print(f"\n✅ 成功加载 {len(templates)} 个模板")
    
    return templates


def preprocess_cell_image(cell_image):
    """
    预处理待匹配的格子图像
    
    参数:
        cell_image: 格子图像
    
    返回:
        gray_cell: 灰度格子图像
    """
    # 转灰度图
    if len(cell_image.shape) == 2:
        # 已经是灰度图
        gray_cell = cell_image
    elif cell_image.shape[2] == 4:
        # RGBA → 灰度
        gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_RGBA2GRAY)
    elif cell_image.shape[2] == 3:
        # RGB → 灰度
        gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_cell = cell_image
    
    return gray_cell


def template_matching(cell_image, templates, piece_color, save_debug=False, output_dir="output"):
    """
    模板匹配（完全复刻参考项目）
    
    参数:
        cell_image: 待匹配的格子图像
        templates: 模板字典
        piece_color: 棋子颜色 ("red" 或 "black")
        save_debug: 是否保存调试图像
        output_dir: 输出目录
    
    返回:
        matched_piece: 匹配到的棋子类型
        max_match_value: 最大匹配值
        all_results: 所有模板的匹配结果
    """
    print("\n" + "="*60)
    print("模板匹配")
    print("="*60)
    
    # 1. 预处理格子图像（转灰度）
    gray_cell = preprocess_cell_image(cell_image)
    cell_size = (gray_cell.shape[1], gray_cell.shape[0])  # (width, height)
    
    print(f"格子图像尺寸: {cell_size[0]}x{cell_size[1]}")
    print(f"棋子颜色: {piece_color}")
    
    # 2. 筛选对应颜色的模板
    color_prefix = f"{piece_color}_"
    color_templates = {name: template for name, template in templates.items() 
                      if name.startswith(color_prefix)}
    
    print(f"筛选后的模板数量: {len(color_templates)}")
    
    if len(color_templates) == 0:
        print(f"❌ 没有找到 {piece_color} 颜色的模板")
        return 'none', 0.0, []
    
    # 3. 遍历所有模板进行匹配
    print(f"\n开始匹配...")
    print("-"*60)
    
    max_match_value = -1
    matched_piece = 'none'
    matched_template_name = None
    all_results = []
    
    for template_name, template in color_templates.items():
        # 3.1 调整模板大小到格子大小
        resized_template = cv2.resize(template, cell_size)
        
        # 3.2 模板匹配（归一化相关系数）
        result = cv2.matchTemplate(gray_cell, resized_template, 
                                   TemplateMatchingConfig.MATCHING_METHOD)
        
        # 3.3 获取最大匹配值
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 3.4 提取棋子类型
        # 模板名称格式: red_rook, black_king 等
        piece_type_name = template_name.split('_')[1] if '_' in template_name else template_name
        piece_type_cn = TemplateMatchingConfig.PIECE_TYPE_MAP.get(piece_type_name, piece_type_name)
        
        print(f"   {template_name:20s} → 匹配度: {max_val:.4f} ({piece_type_cn})")
        
        # 记录结果
        all_results.append({
            'template_name': template_name,
            'piece_type': piece_type_name,
            'piece_type_cn': piece_type_cn,
            'match_value': max_val,
            'resized_template': resized_template.copy()
        })
        
        # 3.5 更新最佳匹配
        if max_val > max_match_value:
            max_match_value = max_val
            matched_piece = piece_type_name
            matched_template_name = template_name
    
    # 4. 阈值判断
    print("-"*60)
    print(f"\n最佳匹配:")
    print(f"   模板: {matched_template_name}")
    print(f"   类型: {matched_piece} ({TemplateMatchingConfig.PIECE_TYPE_MAP.get(matched_piece, matched_piece)})")
    print(f"   匹配度: {max_match_value:.4f}")
    print(f"   阈值: {TemplateMatchingConfig.MATCHING_THRESHOLD}")
    
    if max_match_value > TemplateMatchingConfig.MATCHING_THRESHOLD:
        print(f"   ✅ 匹配成功！")
        final_result = matched_piece
    else:
        print(f"   ❌ 匹配度低于阈值，识别失败")
        final_result = 'none'
    
    # 5. 保存调试图像
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存灰度格子图像
        cv2.imwrite(os.path.join(output_dir, "01_gray_cell.png"), gray_cell)
        
        # 保存所有匹配结果（按匹配度排序）
        all_results.sort(key=lambda x: x['match_value'], reverse=True)
        
        for i, result in enumerate(all_results[:5]):  # 只保存前5个
            template_img = result['resized_template']
            match_val = result['match_value']
            template_name = result['template_name']
            
            # 创建对比图像
            h = max(gray_cell.shape[0], template_img.shape[0])
            w = gray_cell.shape[1] + template_img.shape[1] + 10
            comparison = np.ones((h, w), dtype=np.uint8) * 255
            
            comparison[0:gray_cell.shape[0], 0:gray_cell.shape[1]] = gray_cell
            comparison[0:template_img.shape[0], gray_cell.shape[1]+10:] = template_img
            
            # 添加文字
            text = f"{template_name}: {match_val:.4f}"
            cv2.putText(comparison, text, (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            filename = f"02_match_{i+1}_{template_name}_{match_val:.4f}.png"
            cv2.imwrite(os.path.join(output_dir, filename), comparison)
        
        print(f"\n💾 已保存调试图像到: {output_dir}")
    
    return final_result, max_match_value, all_results


def main():
    """主函数"""
    print("="*60)
    print("步骤4：模板匹配")
    print("="*60)
    
    # 1. 加载所有模板
    templates = load_all_templates(TEMPLATE_DIR)
    
    if len(templates) == 0:
        print("\n❌ 没有加载到任何模板，请检查模板目录")
        return
    
    # 2. 读取待匹配图像
    print(f"\n📂 待匹配图像: {INPUT_IMAGE}")
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 文件不存在: {INPUT_IMAGE}")
        return
    
    cell_image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
    
    if cell_image is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    
    print(f"✓ 图像尺寸: {cell_image.shape[1]}x{cell_image.shape[0]}")
    
    # 3. 模板匹配
    matched_piece, match_value, all_results = template_matching(
        cell_image, templates, PIECE_COLOR, SAVE_DEBUG, OUTPUT_DIR
    )
    
    # 4. 输出结果
    print("\n" + "="*60)
    print("✅ 匹配完成！")
    print("="*60)
    print(f"识别结果: {matched_piece}")
    if matched_piece != 'none':
        piece_cn = TemplateMatchingConfig.PIECE_TYPE_MAP.get(matched_piece, matched_piece)
        print(f"棋子名称: {piece_cn}")
    print(f"匹配度: {match_value:.4f}")
    
    if SAVE_DEBUG:
        print(f"\n📁 调试文件:")
        print(f"   {OUTPUT_DIR}/01_gray_cell.png           - 灰度格子图像")
        print(f"   {OUTPUT_DIR}/02_match_1_*.png          - 最佳匹配对比")
        print(f"   {OUTPUT_DIR}/02_match_2_*.png          - 第2匹配对比")
        print(f"   ...")


if __name__ == "__main__":
    main()

