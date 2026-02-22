"""
步骤1改进版：基于第一根横线的精确分割
保留原有的霍夫变换+KMeans检测，但改用第一根横线定位棋盘
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import ImageGrab
import time

# ============ 配置参数 ============
class ChessboardConfig:
    # 图像处理参数
    GAUSSIAN_BLUR_SIZE = 5
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 80
    MIN_LINE_LENGTH = 100
    MAX_LINE_GAP = 50
    
    # 角度阈值
    HORIZONTAL_ANGLE_THRESHOLD = 10
    VERTICAL_ANGLE_THRESHOLD = 80
    
    # 棋盘比例（高/宽）
    BOARD_HEIGHT_RATIO = 10.0 / 9.0  # 10行9列，高度是宽度的10/9
    
    # 边界扩展（像素）- 用于微调棋盘边界
    # 正数表示向外扩展，负数表示向内收缩
    EXPAND_TOP = 0      # 向上扩展（负数表示向下收缩）
    EXPAND_BOTTOM = -3   # 向下扩展（负数表示向上收缩）
    EXPAND_LEFT = 0     # 向左扩展（负数表示向右收缩）
    EXPAND_RIGHT = 0   # 向右扩展（负数表示向左收缩）

# ============ 用户配置 ============
USE_REGION_SELECT = True
SAVE_DEBUG_IMAGES = True
# =================================


class RegionSelector:
    """鼠标框选区域工具"""
    
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.screenshot = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.selecting = False
    
    def select_region(self):
        print("📸 正在截取全屏...")
        screenshot = ImageGrab.grab()
        self.screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        h, w = self.screenshot.shape[:2]
        scale = min(1.0, 1200 / w, 800 / h)
        display_w = int(w * scale)
        display_h = int(h * scale)
        display_img = cv2.resize(self.screenshot, (display_w, display_h))
        
        print("🖱️  请在窗口中框选棋盘区域...")
        print("   - 按住鼠标左键拖动选择区域")
        print("   - 按 Enter 确认")
        print("   - 按 ESC 取消")
        
        cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Select Region", cv2.WND_PROP_TOPMOST, 1)  # 窗口置顶
        cv2.setMouseCallback("Select Region", self.mouse_callback)
        
        while True:
            temp_img = display_img.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(temp_img, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow("Select Region", temp_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        if not self.start_point or not self.end_point:
            return None
        
        x1 = int(min(self.start_point[0], self.end_point[0]) / scale)
        y1 = int(min(self.start_point[1], self.end_point[1]) / scale)
        x2 = int(max(self.start_point[0], self.end_point[0]) / scale)
        y2 = int(max(self.start_point[1], self.end_point[1]) / scale)
        
        region = self.screenshot[y1:y2, x1:x2]
        print(f"✅ 已选择区域: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"📐 区域尺寸: {x2-x1} x {y2-y1}")
        
        return region


def detect_top_border(cropped_region, save_debug=False):
    """
    检测棋盘上边框（第一根横线）
    返回：第一根横线的左右两点坐标
    """
    # 1. 灰度化和模糊
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ChessboardConfig.GAUSSIAN_BLUR_SIZE, 
                                      ChessboardConfig.GAUSSIAN_BLUR_SIZE), 0)
    
    # 2. 边缘检测
    edges = cv2.Canny(blurred, ChessboardConfig.CANNY_THRESHOLD1, 
                     ChessboardConfig.CANNY_THRESHOLD2)
    
    if save_debug:
        cv2.imwrite("debug_02_edges.png", edges)
    
    # 3. 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           ChessboardConfig.HOUGH_THRESHOLD,
                           ChessboardConfig.MIN_LINE_LENGTH,
                           ChessboardConfig.MAX_LINE_GAP)
    
    if lines is None or len(lines) == 0:
        print("❌ 无法检测到棋盘线条")
        return None
    
    # 4. 筛选水平线
    horizontal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        if abs(angle) < ChessboardConfig.HORIZONTAL_ANGLE_THRESHOLD:
            horizontal_lines.append([x1, y1, x2, y2])
    
    print(f"   检测到 {len(horizontal_lines)} 条水平线")
    
    if len(horizontal_lines) == 0:
        print("❌ 未检测到水平线")
        return None
    
    # 5. 找到最上面的横线（Y坐标最小）
    # 对每条线取平均Y坐标
    top_line = min(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)
    
    x1, y1, x2, y2 = top_line
    
    # 确保左点在右点左边
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    
    avg_y = (y1 + y2) / 2
    
    print(f"   第一根横线（上边框）:")
    print(f"     左点: ({x1}, {y1})")
    print(f"     右点: ({x2}, {y2})")
    print(f"     平均Y: {avg_y:.1f}")
    print(f"     宽度: {x2 - x1}")
    
    return {
        'left_x': x1,
        'left_y': y1,
        'right_x': x2,
        'right_y': y2,
        'avg_y': avg_y,
        'width': x2 - x1
    }


def segment_chessboard_new(cropped_region, save_debug=False):
    """
    新方法：基于上边框精确分割
    1. 检测上边框（第一根横线）
    2. 根据宽度向下取 10:9 矩形
    3. 应用边界扩展
    4. 均匀分割成 90 个格子
    """
    print("\n步骤2：检测上边框")
    print("-"*60)
    
    # 1. 检测上边框（第一根横线）
    top_border = detect_top_border(cropped_region, save_debug)
    
    if top_border is None:
        return [], {}, []
    
    # 2. 根据上边框定位棋盘（10:9矩形）
    print(f"\n步骤3：基于上边框定位棋盘（10:9矩形）")
    print("-"*60)
    
    # 使用上边框的宽度和左端点
    board_x = top_border['left_x']
    board_y = int(top_border['avg_y'])
    board_width = top_border['width']
    board_height = int(board_width * ChessboardConfig.BOARD_HEIGHT_RATIO)
    
    print(f"   初始起点: ({board_x}, {board_y})")
    print(f"   初始宽度: {board_width}")
    print(f"   初始高度: {board_height} (宽度 × 10/9)")
    print(f"   宽高比: {board_height / board_width:.4f}")
    
    # 3. 裁剪10:9矩形区域
    chessboard_rect = cropped_region[board_y:board_y+board_height, 
                                     board_x:board_x+board_width]
    
    if save_debug:
        cv2.imwrite("debug_03_chessboard.png", chessboard_rect)
        print(f"   💾 已保存: debug_03_chessboard.png (10:9矩形)")
    
    # 4. 应用边界扩展（用于分割格子）
    print(f"\n步骤4：应用边界扩展")
    print("-"*60)
    
    expand_top = ChessboardConfig.EXPAND_TOP
    expand_bottom = ChessboardConfig.EXPAND_BOTTOM
    expand_left = ChessboardConfig.EXPAND_LEFT
    expand_right = ChessboardConfig.EXPAND_RIGHT
    
    print(f"   扩展参数: 上={expand_top}px, 下={expand_bottom}px, 左={expand_left}px, 右={expand_right}px")
    
    # 计算扩展后的区域
    # 正数表示向外扩展，负数表示向内收缩
    expanded_x = board_x - expand_left          # 左边：减去expand_left（正数向左扩展，负数向右收缩）
    expanded_y = board_y - expand_top           # 上边：减去expand_top（正数向上扩展，负数向下收缩）
    expanded_width = board_width + expand_left + expand_right    # 宽度增加
    expanded_height = board_height + expand_top + expand_bottom  # 高度增加
    
    # 边界检查
    expanded_x = max(0, expanded_x)
    expanded_y = max(0, expanded_y)
    
    if expanded_y + expanded_height > cropped_region.shape[0]:
        expanded_height = cropped_region.shape[0] - expanded_y
        print(f"   ⚠️  高度超出边界，调整为: {expanded_height}")
    
    if expanded_x + expanded_width > cropped_region.shape[1]:
        expanded_width = cropped_region.shape[1] - expanded_x
        print(f"   ⚠️  宽度超出边界，调整为: {expanded_width}")
    
    print(f"   扩展后起点: ({expanded_x}, {expanded_y})")
    print(f"   扩展后宽度: {expanded_width}")
    print(f"   扩展后高度: {expanded_height}")
    
    # 5. 裁剪扩展后的棋盘区域
    expanded_board = cropped_region[expanded_y:expanded_y+expanded_height, 
                                    expanded_x:expanded_x+expanded_width]
    
    if save_debug:
        cv2.imwrite("debug_03_expanded_board.png", expanded_board)
        print(f"   💾 已保存: debug_03_expanded_board.png (扩展后用于分割)")
    
    # 6. 均匀分割 10行 x 9列
    print(f"\n步骤5：均匀分割90个格子")
    print("-"*60)
    
    rows, cols = 10, 9
    cell_height = expanded_height // rows
    cell_width = expanded_width // cols
    
    print(f"   格子尺寸: {cell_width} x {cell_height}")
    
    grid_cells = []
    
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height if i < rows - 1 else expanded_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width if j < cols - 1 else expanded_width
            
            cell = expanded_board[y1:y2, x1:x2]
            grid_cells.append(cell)
    
    expanded_rect = {
        'x': expanded_x,
        'y': expanded_y,
        'width': expanded_width,
        'height': expanded_height
    }
    
    # 生成均匀的格子高度列表（用于兼容）
    grid_heights = [cell_height] * rows
    
    return grid_cells, expanded_rect, grid_heights


def detect_and_extract_chessboard(image, save_debug=False):
    """
    检测并提取棋盘（主函数）
    """
    print("\n" + "="*60)
    print("步骤1：检测棋盘边框")
    print("="*60)
    
    # 1. 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯模糊
    blurred = cv2.GaussianBlur(gray, (ChessboardConfig.GAUSSIAN_BLUR_SIZE,
                                      ChessboardConfig.GAUSSIAN_BLUR_SIZE), 0)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blurred, ChessboardConfig.CANNY_THRESHOLD1,
                     ChessboardConfig.CANNY_THRESHOLD2)
    
    if save_debug:
        cv2.imwrite("debug_01_edges.png", edges)
    
    # 4. 形态学操作（膨胀+腐蚀）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    final_edges = cv2.erode(dilated, kernel, iterations=1)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("❌ 未检测到任何轮廓")
        return [], {}, []
    
    # 6. 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    print(f"✓ 检测到最大轮廓: ({x}, {y}, {w}, {h})")
    print(f"✓ 轮廓宽高比: {h/w:.3f}")
    
    # 7. 裁剪轮廓区域
    cropped_region = image[y:y+h, x:x+w]
    
    # 8. 使用新方法分割棋盘
    grid_cells, expanded_rect, grid_heights = segment_chessboard_new(
        cropped_region, save_debug
    )
    
    if len(grid_cells) == 0:
        return [], {}, []
    
    # 9. 计算最终棋盘位置
    chessboard_rect = {
        'x': x + expanded_rect['x'],
        'y': y + expanded_rect['y'],
        'width': expanded_rect['width'],
        'height': expanded_rect['height']
    }
    
    print(f"✓ 分割完成: {len(grid_cells)} 个格子")
    print(f"✓ 最终棋盘位置: {chessboard_rect}")
    
    return grid_cells, chessboard_rect, grid_heights


def main():
    """主函数"""
    print("="*60)
    print("棋盘检测测试 - 基于第一根横线的精确分割")
    print("="*60)
    
    # 获取图像
    if USE_REGION_SELECT:
        selector = RegionSelector()
        time.sleep(2)
        image = selector.select_region()
        if image is None:
            print("❌ 未选择区域")
            return
    else:
        print("📸 将在2秒后截取全屏...")
        time.sleep(2)
        screenshot = ImageGrab.grab()
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # 保存原始图像
    if SAVE_DEBUG_IMAGES:
        cv2.imwrite("debug_00_original.png", image)
        print("💾 已保存原始图像: debug_00_original.png")
    
    # 检测棋盘
    grid_cells, chessboard_rect, grid_heights = detect_and_extract_chessboard(
        image, SAVE_DEBUG_IMAGES
    )
    
    if len(grid_cells) == 0:
        print("\n❌ 棋盘检测失败")
        return
    
    # 保存所有90个格子
    if SAVE_DEBUG_IMAGES:
        print("\n💾 保存所有格子...")
        import os
        os.makedirs("step1_output", exist_ok=True)
        
        for row in range(10):
            for col in range(9):
                idx = row * 9 + col
                if idx < len(grid_cells):
                    filename = f"step1_output/cell_{row+1:02d}-{col+1:02d}.png"
                    cv2.imwrite(filename, grid_cells[idx])
        
        print(f"   ✓ 已保存 {len(grid_cells)} 个格子到 step1_output/")
        print(f"   格子命名: cell_01-01.png ~ cell_10-09.png")
    
    print("\n" + "="*60)
    print("✅ 棋盘检测完成！")
    print("="*60)
    print(f"📊 统计信息:")
    print(f"   格子数量: {len(grid_cells)}")
    print(f"   棋盘位置: ({chessboard_rect['x']}, {chessboard_rect['y']})")
    print(f"   棋盘尺寸: {chessboard_rect['width']} x {chessboard_rect['height']}")
    print(f"   宽高比: {chessboard_rect['height'] / chessboard_rect['width']:.3f}")
    print(f"   格子高度: {grid_heights}")


if __name__ == "__main__":
    main()

