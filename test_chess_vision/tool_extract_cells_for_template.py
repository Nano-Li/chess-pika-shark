"""
工具脚本：提取棋盘格子用于制作模板
从给定的棋盘图片中检测棋盘，分割为90个格子，保存到指定文件夹
"""

import cv2
import numpy as np
import os
from datetime import datetime


# ============ 用户配置 ============
# 输入图像路径
INPUT_IMAGE = r"D:\Projects\AIchess\chess_board_phone.png"

# 输出目录（自动添加时间戳）
OUTPUT_BASE_DIR = "template_cells"

# 棋盘检测参数
class ChessboardConfig:
    # Canny边缘检测参数
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    
    # Hough直线检测参数
    HOUGH_THRESHOLD = 100
    HOUGH_MIN_LINE_LENGTH = 200
    HOUGH_MAX_LINE_GAP = 20
    
    # 顶边检测参数
    TOP_EDGE_Y_MAX = 200  # 顶边必须在图像上方200像素内
    HORIZONTAL_ANGLE_THRESHOLD = 5  # 水平线角度阈值（度）
    
    # 棋盘尺寸
    BOARD_RATIO_WIDTH = 9   # 宽度比例
    BOARD_RATIO_HEIGHT = 10  # 高度比例
    
    # 边界微调（像素）
    EXPAND_TOP = -10
    EXPAND_BOTTOM = 0
    EXPAND_LEFT = 0
    EXPAND_RIGHT = 0
    
    # 格子分割
    GRID_ROWS = 10
    GRID_COLS = 9

# =================================


def detect_top_edge(image, config):
    """
    检测棋盘顶边（第一条水平线）
    返回: (left_point, right_point) 或 None
    """
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测
    edges = cv2.Canny(gray, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)
    
    # Hough直线检测
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP
    )
    
    if lines is None:
        return None, edges
    
    # 筛选水平线（顶边候选）
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算角度
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        
        # 筛选条件：
        # 1. 接近水平（角度小于阈值）
        # 2. 在图像上方区域
        if angle < config.HORIZONTAL_ANGLE_THRESHOLD and y1 < config.TOP_EDGE_Y_MAX:
            # 计算线段中点的y坐标
            mid_y = (y1 + y2) / 2
            horizontal_lines.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'mid_y': mid_y,
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
            })
    
    if not horizontal_lines:
        return None, edges
    
    # 按y坐标排序，取最上方的线
    horizontal_lines.sort(key=lambda l: l['mid_y'])
    top_line = horizontal_lines[0]
    
    # 返回左右端点
    left_point = (top_line['x1'], top_line['y1'])
    right_point = (top_line['x2'], top_line['y2'])
    
    # 确保left在右边，right在左边
    if left_point[0] > right_point[0]:
        left_point, right_point = right_point, left_point
    
    return (left_point, right_point), edges


def extract_chessboard(image, top_edge, config):
    """
    根据顶边提取棋盘（10:9矩形）
    返回: 棋盘图像
    """
    left_point, right_point = top_edge
    
    # 计算棋盘宽度
    board_width = right_point[0] - left_point[0]
    
    # 计算棋盘高度（10:9比例）
    board_height = int(board_width * config.BOARD_RATIO_HEIGHT / config.BOARD_RATIO_WIDTH)
    
    # 棋盘左上角坐标
    board_x = left_point[0]
    board_y = left_point[1]
    
    # 裁剪棋盘（10:9矩形）
    chessboard = image[
        board_y : board_y + board_height,
        board_x : board_x + board_width
    ]
    
    return chessboard, (board_x, board_y, board_width, board_height)


def expand_chessboard(image, chessboard, board_rect, config):
    """
    应用边界微调
    返回: 扩展后的棋盘图像
    """
    board_x, board_y, board_width, board_height = board_rect
    
    # 计算扩展后的坐标
    new_x = board_x - config.EXPAND_LEFT
    new_y = board_y - config.EXPAND_TOP
    new_width = board_width + config.EXPAND_LEFT + config.EXPAND_RIGHT
    new_height = board_height + config.EXPAND_TOP + config.EXPAND_BOTTOM
    
    # 边界检查
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_width = min(new_width, image.shape[1] - new_x)
    new_height = min(new_height, image.shape[0] - new_y)
    
    # 裁剪扩展后的棋盘
    expanded_board = image[
        new_y : new_y + new_height,
        new_x : new_x + new_width
    ]
    
    return expanded_board, (new_x, new_y, new_width, new_height)


def split_into_cells(chessboard, config):
    """
    将棋盘均匀分割为90个格子
    返回: 格子图像列表 [(row, col, cell_image), ...]
    """
    board_height, board_width = chessboard.shape[:2]
    
    # 计算每个格子的尺寸
    cell_width = board_width / config.GRID_COLS
    cell_height = board_height / config.GRID_ROWS
    
    cells = []
    
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            # 计算格子坐标
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            
            # 裁剪格子
            cell = chessboard[y1:y2, x1:x2]
            
            cells.append((row, col, cell))
    
    return cells


def save_debug_images(image, edges, top_edge, chessboard, expanded_board, output_dir):
    """
    保存调试图像
    """
    # 1. 保存边缘检测结果
    cv2.imwrite(os.path.join(output_dir, "debug_01_edges.png"), edges)
    
    # 2. 保存顶边检测结果
    if top_edge:
        debug_top_edge = image.copy()
        left_point, right_point = top_edge
        cv2.line(debug_top_edge, left_point, right_point, (0, 255, 0), 3)
        cv2.circle(debug_top_edge, left_point, 8, (255, 0, 0), -1)
        cv2.circle(debug_top_edge, right_point, 8, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "debug_02_top_edge.png"), debug_top_edge)
    
    # 3. 保存10:9矩形棋盘
    cv2.imwrite(os.path.join(output_dir, "debug_03_chessboard.png"), chessboard)
    
    # 4. 保存扩展后的棋盘
    cv2.imwrite(os.path.join(output_dir, "debug_04_expanded_board.png"), expanded_board)
    
    # 5. 保存带网格线的棋盘
    debug_grid = expanded_board.copy()
    board_height, board_width = expanded_board.shape[:2]
    cell_width = board_width / ChessboardConfig.GRID_COLS
    cell_height = board_height / ChessboardConfig.GRID_ROWS
    
    # 绘制垂直线
    for col in range(ChessboardConfig.GRID_COLS + 1):
        x = int(col * cell_width)
        cv2.line(debug_grid, (x, 0), (x, board_height), (0, 255, 0), 1)
    
    # 绘制水平线
    for row in range(ChessboardConfig.GRID_ROWS + 1):
        y = int(row * cell_height)
        cv2.line(debug_grid, (0, y), (board_width, y), (0, 255, 0), 1)
    
    cv2.imwrite(os.path.join(output_dir, "debug_05_grid.png"), debug_grid)


def main():
    """主函数"""
    print("="*60)
    print("工具脚本：提取棋盘格子用于制作模板")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 输入文件不存在: {INPUT_IMAGE}")
        return
    
    print(f"\n📂 输入图像: {INPUT_IMAGE}")
    
    # 读取图像
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    
    print(f"📐 图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 创建输出目录
    output_dir = OUTPUT_BASE_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 输出目录: {output_dir}")
    
    # 步骤1：检测顶边
    print("\n" + "-"*60)
    print("步骤1：检测棋盘顶边")
    print("-"*60)
    
    top_edge, edges = detect_top_edge(image, ChessboardConfig)
    
    if top_edge is None:
        print("❌ 未检测到顶边")
        return
    
    left_point, right_point = top_edge
    print(f"✓ 检测到顶边:")
    print(f"  左端点: {left_point}")
    print(f"  右端点: {right_point}")
    print(f"  宽度: {right_point[0] - left_point[0]} 像素")
    
    # 步骤2：提取棋盘（10:9矩形）
    print("\n" + "-"*60)
    print("步骤2：提取棋盘（10:9矩形）")
    print("-"*60)
    
    chessboard, board_rect = extract_chessboard(image, top_edge, ChessboardConfig)
    board_x, board_y, board_width, board_height = board_rect
    
    print(f"✓ 棋盘矩形:")
    print(f"  位置: ({board_x}, {board_y})")
    print(f"  尺寸: {board_width}x{board_height}")
    print(f"  比例: {board_width/board_height:.3f} (理论值: {ChessboardConfig.BOARD_RATIO_WIDTH/ChessboardConfig.BOARD_RATIO_HEIGHT:.3f})")
    
    # 步骤3：应用边界微调
    print("\n" + "-"*60)
    print("步骤3：应用边界微调")
    print("-"*60)
    
    expanded_board, expanded_rect = expand_chessboard(image, chessboard, board_rect, ChessboardConfig)
    new_x, new_y, new_width, new_height = expanded_rect
    
    print(f"✓ 边界微调:")
    print(f"  上: {ChessboardConfig.EXPAND_TOP:+d} 像素")
    print(f"  下: {ChessboardConfig.EXPAND_BOTTOM:+d} 像素")
    print(f"  左: {ChessboardConfig.EXPAND_LEFT:+d} 像素")
    print(f"  右: {ChessboardConfig.EXPAND_RIGHT:+d} 像素")
    print(f"✓ 扩展后棋盘:")
    print(f"  位置: ({new_x}, {new_y})")
    print(f"  尺寸: {new_width}x{new_height}")
    
    # 步骤4：分割为90个格子
    print("\n" + "-"*60)
    print("步骤4：分割为90个格子")
    print("-"*60)
    
    cells = split_into_cells(expanded_board, ChessboardConfig)
    
    print(f"✓ 分割完成: {len(cells)} 个格子")
    print(f"  格子尺寸: {cells[0][2].shape[1]}x{cells[0][2].shape[0]} 像素")
    
    # 步骤5：保存格子图像
    print("\n" + "-"*60)
    print("步骤5：保存格子图像")
    print("-"*60)
    
    cells_dir = os.path.join(output_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    
    for row, col, cell in cells:
        # 格子命名：cell_行号-列号.png（行列从1开始）
        filename = f"cell_{row+1:02d}-{col+1:02d}.png"
        filepath = os.path.join(cells_dir, filename)
        cv2.imwrite(filepath, cell)
    
    print(f"✓ 已保存 {len(cells)} 个格子到: {cells_dir}")
    
    # 步骤6：保存调试图像
    print("\n" + "-"*60)
    print("步骤6：保存调试图像")
    print("-"*60)
    
    save_debug_images(image, edges, top_edge, chessboard, expanded_board, output_dir)
    
    print(f"✓ 已保存调试图像:")
    print(f"  - debug_01_edges.png (边缘检测)")
    print(f"  - debug_02_top_edge.png (顶边检测)")
    print(f"  - debug_03_chessboard.png (10:9矩形)")
    print(f"  - debug_04_expanded_board.png (边界微调后)")
    print(f"  - debug_05_grid.png (网格线)")
    
    # 完成
    print("\n" + "="*60)
    print("✅ 提取完成！")
    print("="*60)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"📁 格子目录: {cells_dir}")
    print(f"\n💡 下一步:")
    print(f"   1. 查看 debug_05_grid.png 确认格子分割正确")
    print(f"   2. 从 cells/ 文件夹中选择合适的格子制作模板")
    print(f"   3. 使用 convert_to_bw.py 将选中的格子转换为黑白模板")


if __name__ == "__main__":
    main()

