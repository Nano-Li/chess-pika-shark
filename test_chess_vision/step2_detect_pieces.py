"""
步骤2：检测棋子和颜色
完全复刻参考项目的方法：
1. 检测格子中是否有棋子（对比度检测）
2. 检测棋子颜色（HSV颜色空间）
3. 在棋盘图像上标注结果
"""

import cv2
import numpy as np
import os
import glob

# ============ 配置参数 ============
class PieceDetectionConfig:
    # 对比度阈值（用于检测是否有棋子）
    CONTRAST_THRESHOLD = 30
    
    # 红色HSV范围（用于颜色检测）
    RED_HSV_LOWER_1 = (0, 120, 120)
    RED_HSV_UPPER_1 = (10, 255, 255)
    RED_HSV_LOWER_2 = (160, 120, 120)
    RED_HSV_UPPER_2 = (179, 255, 255)
    
    # 黑色HSV范围
    BLACK_HSV_LOWER = (0, 0, 0)
    BLACK_HSV_UPPER = (180, 255, 80)
    
    # 颜色像素比例阈值（优化后的判断逻辑）
    RED_RATIO_THRESHOLD = 0.05    # 红色比例阈值
    BLACK_RATIO_THRESHOLD = 0.03  # 黑色比例阈值（单独设置，更低）

# ============ 用户配置 ============
# 输入目录（step1 输出的格子图像）
INPUT_DIR = "step1_output"

# 棋盘图像（用于标注）
BOARD_IMAGE = "debug_03_expanded_board.png"

# 输出目录
OUTPUT_DIR = "step2_output"

# 是否保存调试信息
SAVE_DEBUG = True
# =================================


def detect_piece_in_cell(cell_image, contrast_threshold=30):
    """
    检测格子中是否有棋子（基于对比度）
    完全复刻参考项目的方法
    
    参数:
        cell_image: 格子图像（BGR）
        contrast_threshold: 对比度阈值
    
    返回:
        has_piece: 是否有棋子
        contrast: 对比度值
    """
    # 1. 转灰度图
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # 2. 计算均值和标准差
    mean, stddev = cv2.meanStdDev(gray)
    
    # 3. 标准差即为对比度
    contrast = stddev[0][0]
    
    # 4. 判断
    has_piece = contrast > contrast_threshold
    
    return has_piece, contrast


def detect_piece_color(cell_image):
    """
    检测棋子颜色（基于HSV颜色空间）
    优化后的判断逻辑
    
    参数:
        cell_image: 格子图像（BGR）
    
    返回:
        color: 'red', 'black', 或 'none'
        red_ratio: 红色像素比例
        black_ratio: 黑色像素比例
    """
    # 1. 转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    
    # 2. 创建红色掩码（两个范围）
    red_mask1 = cv2.inRange(hsv_image, 
                           PieceDetectionConfig.RED_HSV_LOWER_1,
                           PieceDetectionConfig.RED_HSV_UPPER_1)
    red_mask2 = cv2.inRange(hsv_image,
                           PieceDetectionConfig.RED_HSV_LOWER_2,
                           PieceDetectionConfig.RED_HSV_UPPER_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # 3. 创建黑色掩码
    black_mask = cv2.inRange(hsv_image,
                            PieceDetectionConfig.BLACK_HSV_LOWER,
                            PieceDetectionConfig.BLACK_HSV_UPPER)
    
    # 4. 计算像素比例
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    red_pixels = cv2.countNonZero(red_mask)
    black_pixels = cv2.countNonZero(black_mask)
    
    red_ratio = red_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    
    # 5. 优化后的判断逻辑
    # 红色棋子：红色比例 >= 0.05 且 黑色比例 < 0.03
    if red_ratio >= PieceDetectionConfig.RED_RATIO_THRESHOLD and \
       black_ratio < PieceDetectionConfig.BLACK_RATIO_THRESHOLD:
        return 'red', red_ratio, black_ratio
    
    # 黑色棋子：红色比例 < 0.05 且 黑色比例 >= 0.03
    elif red_ratio < PieceDetectionConfig.RED_RATIO_THRESHOLD and \
         black_ratio >= PieceDetectionConfig.BLACK_RATIO_THRESHOLD:
        return 'black', red_ratio, black_ratio
    
    # 其他情况：没有棋子（楚河汉界等）
    else:
        return 'none', red_ratio, black_ratio


def load_all_cells(input_dir):
    """
    加载所有格子图像
    
    参数:
        input_dir: 格子图像目录
    
    返回:
        cells: 字典 {(row, col): cell_image}
    """
    print("\n" + "="*60)
    print("加载格子图像")
    print("="*60)
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return {}
    
    # 查找所有格子图像
    cell_files = glob.glob(os.path.join(input_dir, "cell_*.png"))
    
    if len(cell_files) == 0:
        print(f"❌ 未找到格子图像: {input_dir}/cell_*.png")
        return {}
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📊 找到 {len(cell_files)} 个格子图像")
    
    cells = {}
    
    for cell_file in sorted(cell_files):
        # 解析文件名: cell_01-01.png → (1, 1)
        filename = os.path.basename(cell_file)
        parts = filename.replace('.png', '').split('_')[1].split('-')
        row = int(parts[0])
        col = int(parts[1])
        
        # 读取图像
        cell_image = cv2.imread(cell_file)
        if cell_image is not None:
            cells[(row, col)] = cell_image
    
    print(f"✅ 成功加载 {len(cells)} 个格子")
    
    return cells


def detect_all_pieces(cells, save_debug=False, output_dir="output"):
    """
    检测所有格子中的棋子
    优化后的逻辑：先算对比度，对比度高的再算颜色
    
    参数:
        cells: 格子字典 {(row, col): cell_image}
        save_debug: 是否保存调试信息
        output_dir: 输出目录
    
    返回:
        results: 检测结果列表
    """
    print("\n" + "="*60)
    print("检测棋子和颜色")
    print("="*60)
    
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f"\n对比度阈值: {PieceDetectionConfig.CONTRAST_THRESHOLD}")
    print(f"红色比例阈值: {PieceDetectionConfig.RED_RATIO_THRESHOLD}")
    print(f"黑色比例阈值: {PieceDetectionConfig.BLACK_RATIO_THRESHOLD}")
    print(f"\n优化逻辑:")
    print(f"  1. 对比度 < 30 → 没有棋子")
    print(f"  2. 对比度 >= 30 → 可能有棋子，进入颜色判断:")
    print(f"     - 红色比例 >= 0.05 且 黑色比例 < 0.03 → 红色棋子")
    print(f"     - 红色比例 < 0.05 且 黑色比例 >= 0.03 → 黑色棋子")
    print(f"     - 其他情况 → 没有棋子（楚河汉界等）")
    print(f"\n开始检测...")
    print("-"*60)
    
    piece_count = 0
    red_count = 0
    black_count = 0
    high_contrast_no_piece = 0  # 对比度高但没有棋子（楚河汉界）
    
    for row in range(1, 11):  # 1-10
        for col in range(1, 10):  # 1-9
            if (row, col) not in cells:
                continue
            
            cell_image = cells[(row, col)]
            
            # 步骤1：先检测对比度（节省计算资源）
            has_piece_by_contrast, contrast = detect_piece_in_cell(
                cell_image, 
                PieceDetectionConfig.CONTRAST_THRESHOLD
            )
            
            # 步骤2：如果对比度高，再检测颜色
            if has_piece_by_contrast:
                color, red_ratio, black_ratio = detect_piece_color(cell_image)
                
                # 根据颜色判断是否真的有棋子
                if color == 'red':
                    has_piece = True
                    piece_count += 1
                    red_count += 1
                    marker = 'r'
                    print(f"   [{row:2d},{col:2d}] 红色棋子 | 对比度={contrast:5.1f} | "
                          f"红={red_ratio:.3f} 黑={black_ratio:.3f}")
                elif color == 'black':
                    has_piece = True
                    piece_count += 1
                    black_count += 1
                    marker = 'b'
                    print(f"   [{row:2d},{col:2d}] 黑色棋子 | 对比度={contrast:5.1f} | "
                          f"红={red_ratio:.3f} 黑={black_ratio:.3f}")
                else:
                    # 对比度高但颜色不符合（楚河汉界）
                    has_piece = False
                    marker = ''
                    high_contrast_no_piece += 1
                    # 不打印，减少输出
            else:
                # 对比度低，没有棋子
                has_piece = False
                color = 'none'
                marker = ''
                red_ratio = 0
                black_ratio = 0
            
            results.append({
                'row': row,
                'col': col,
                'has_piece': has_piece,
                'contrast': contrast,
                'color': color,
                'marker': marker,
                'red_ratio': red_ratio,
                'black_ratio': black_ratio
            })
    
    print("-"*60)
    print(f"\n检测统计:")
    print(f"   总格子数: {len(cells)}")
    print(f"   有棋子: {piece_count}")
    print(f"     红色棋子: {red_count}")
    print(f"     黑色棋子: {black_count}")
    print(f"   没有棋子: {len(cells) - piece_count}")
    print(f"     对比度低: {len(cells) - piece_count - high_contrast_no_piece}")
    print(f"     对比度高但无棋子（楚河汉界等）: {high_contrast_no_piece}")
    
    return results


def annotate_board(board_image_path, results, output_path):
    """
    在棋盘图像上标注检测结果
    
    参数:
        board_image_path: 棋盘图像路径
        results: 检测结果列表
        output_path: 输出图像路径
    """
    print("\n" + "="*60)
    print("标注棋盘图像")
    print("="*60)
    
    # 读取棋盘图像
    board = cv2.imread(board_image_path)
    
    if board is None:
        print(f"❌ 无法读取棋盘图像: {board_image_path}")
        return
    
    print(f"📂 棋盘图像: {board_image_path}")
    print(f"📐 图像尺寸: {board.shape[1]}x{board.shape[0]}")
    
    # 计算格子尺寸
    rows, cols = 10, 9
    cell_height = board.shape[0] // rows
    cell_width = board.shape[1] // cols
    
    print(f"📏 格子尺寸: {cell_width}x{cell_height}")
    
    # 标注每个格子
    for result in results:
        if result['marker']:
            row = result['row']
            col = result['col']
            marker = result['marker']
            
            # 计算格子中心位置
            center_x = int((col - 0.5) * cell_width)
            center_y = int((row - 0.5) * cell_height)
            
            # 绘制标记
            if marker == 'r':
                color = (0, 0, 255)  # 红色
                text = 'r'
            elif marker == 'b':
                color = (0, 0, 0)    # 黑色
                text = 'b'
            else:
                color = (128, 128, 128)  # 灰色
                text = '?'
            
            # 绘制圆形背景
            cv2.circle(board, (center_x, center_y), 12, (255, 255, 255), -1)
            cv2.circle(board, (center_x, center_y), 12, color, 2)
            
            # 绘制文字
            cv2.putText(board, text, (center_x - 6, center_y + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 保存标注后的图像
    cv2.imwrite(output_path, board)
    print(f"\n💾 已保存标注图像: {output_path}")


def save_detection_report(results, output_path):
    """
    保存检测报告
    
    参数:
        results: 检测结果列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("棋子检测报告\n")
        f.write("="*100 + "\n\n")
        
        f.write("检测参数:\n")
        f.write(f"  对比度阈值: {PieceDetectionConfig.CONTRAST_THRESHOLD}\n")
        f.write(f"  红色比例阈值: {PieceDetectionConfig.RED_RATIO_THRESHOLD}\n")
        f.write(f"  黑色比例阈值: {PieceDetectionConfig.BLACK_RATIO_THRESHOLD}\n\n")
        
        f.write("判断逻辑:\n")
        f.write(f"  1. 对比度 < {PieceDetectionConfig.CONTRAST_THRESHOLD} → 没有棋子\n")
        f.write(f"  2. 对比度 >= {PieceDetectionConfig.CONTRAST_THRESHOLD} → 可能有棋子，进入颜色判断:\n")
        f.write(f"     - 红色比例 >= {PieceDetectionConfig.RED_RATIO_THRESHOLD} 且 黑色比例 < {PieceDetectionConfig.BLACK_RATIO_THRESHOLD} → 红色棋子\n")
        f.write(f"     - 红色比例 < {PieceDetectionConfig.RED_RATIO_THRESHOLD} 且 黑色比例 >= {PieceDetectionConfig.BLACK_RATIO_THRESHOLD} → 黑色棋子\n")
        f.write(f"     - 其他情况 → 没有棋子（楚河汉界等）\n\n")
        
        # ========== 第一部分：有棋子的格子 ==========
        f.write("="*100 + "\n")
        f.write("第一部分：有棋子的格子\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'位置':<10} {'有棋子':<8} {'对比度':<10} {'颜色':<10} "
                f"{'红色比例':<12} {'黑色比例':<12} {'标记':<6}\n")
        f.write("-"*100 + "\n")
        
        piece_results = [r for r in results if r['has_piece']]
        for result in piece_results:
            f.write(f"[{result['row']:2d},{result['col']:2d}]    "
                   f"{'是':<8} "
                   f"{result['contrast']:8.2f}  "
                   f"{result['color']:<10} "
                   f"{result['red_ratio']:10.3f}  "
                   f"{result['black_ratio']:10.3f}  "
                   f"{result['marker']:<6}\n")
        
        f.write("-"*100 + "\n")
        f.write(f"小计: {len(piece_results)} 个格子有棋子\n\n")
        
        # ========== 第二部分：所有90个格子 ==========
        f.write("="*100 + "\n")
        f.write("第二部分：所有90个格子的完整数据\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'位置':<10} {'有棋子':<8} {'对比度':<10} {'颜色':<10} "
                f"{'红色比例':<12} {'黑色比例':<12} {'标记':<6}\n")
        f.write("-"*100 + "\n")
        
        for result in results:
            has_piece_str = '是' if result['has_piece'] else '否'
            f.write(f"[{result['row']:2d},{result['col']:2d}]    "
                   f"{has_piece_str:<8} "
                   f"{result['contrast']:8.2f}  "
                   f"{result['color']:<10} "
                   f"{result['red_ratio']:10.3f}  "
                   f"{result['black_ratio']:10.3f}  "
                   f"{result['marker']:<6}\n")
        
        f.write("-"*100 + "\n")
        f.write(f"总计: {len(results)} 个格子\n\n")
        
        # ========== 统计信息 ==========
        f.write("="*100 + "\n")
        f.write("统计信息\n")
        f.write("="*100 + "\n\n")
        
        piece_count = sum(1 for r in results if r['has_piece'])
        red_count = sum(1 for r in results if r['marker'] == 'r')
        black_count = sum(1 for r in results if r['marker'] == 'b')
        unknown_count = sum(1 for r in results if r['marker'] == '?')
        empty_count = len(results) - piece_count
        
        f.write(f"总格子数: {len(results)}\n")
        f.write(f"  有棋子: {piece_count}\n")
        f.write(f"    红色棋子: {red_count}\n")
        f.write(f"    黑色棋子: {black_count}\n")
        f.write(f"    未知颜色: {unknown_count}\n")
        f.write(f"  空格子: {empty_count}\n\n")
        
        # ========== 按行统计（用于分析楚河汉界） ==========
        f.write("="*100 + "\n")
        f.write("按行统计（用于分析楚河汉界）\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'行号':<6} {'有棋子数':<10} {'空格子数':<10} {'平均对比度':<12} "
                f"{'对比度范围':<20}\n")
        f.write("-"*100 + "\n")
        
        for row in range(1, 11):
            row_results = [r for r in results if r['row'] == row]
            row_piece_count = sum(1 for r in row_results if r['has_piece'])
            row_empty_count = len(row_results) - row_piece_count
            row_contrasts = [r['contrast'] for r in row_results]
            avg_contrast = sum(row_contrasts) / len(row_contrasts) if row_contrasts else 0
            min_contrast = min(row_contrasts) if row_contrasts else 0
            max_contrast = max(row_contrasts) if row_contrasts else 0
            
            f.write(f"第{row:2d}行  "
                   f"{row_piece_count:<10} "
                   f"{row_empty_count:<10} "
                   f"{avg_contrast:10.2f}  "
                   f"{min_contrast:6.2f} ~ {max_contrast:6.2f}\n")
        
        f.write("-"*100 + "\n\n")
    
    print(f"💾 已保存检测报告: {output_path}")


def main():
    """主函数"""
    print("="*60)
    print("步骤2：检测棋子和颜色")
    print("="*60)
    
    # 1. 加载所有格子图像
    cells = load_all_cells(INPUT_DIR)
    
    if len(cells) == 0:
        print("\n❌ 没有加载到任何格子图像")
        return
    
    # 2. 检测所有棋子
    results = detect_all_pieces(cells, SAVE_DEBUG, OUTPUT_DIR)
    
    # 3. 标注棋盘图像
    if os.path.exists(BOARD_IMAGE):
        output_image = os.path.join(OUTPUT_DIR, "annotated_board.png")
        annotate_board(BOARD_IMAGE, results, output_image)
    else:
        print(f"\n⚠️  棋盘图像不存在: {BOARD_IMAGE}")
        print(f"   跳过标注步骤")
    
    # 4. 保存检测报告
    if SAVE_DEBUG:
        report_path = os.path.join(OUTPUT_DIR, "detection_report.txt")
        save_detection_report(results, report_path)
    
    print("\n" + "="*60)
    print("✅ 检测完成！")
    print("="*60)
    print(f"\n📁 输出文件:")
    print(f"   {OUTPUT_DIR}/annotated_board.png    - 标注后的棋盘图像")
    print(f"   {OUTPUT_DIR}/detection_report.txt   - 检测报告")


if __name__ == "__main__":
    main()

