"""
步骤1：棋盘检测 - 完全复刻 xiangqi-analysis 项目
detectAndExtractChessboard 函数的Python实现
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import ImageGrab
import time

# ============ 配置参数（对应 ChessboardConfig） ============
class ChessboardConfig:
    # 棋盘检测参数
    EXPAND_RATIO_W = 0.03          # 棋盘区域水平扩展比例
    EXPAND_RATIO_H = 0.03           # 棋盘区域垂直扩展比例（基础值）
    BOTTOM_OFFSET_RATIO = 0.00      # 底部偏移比例（基础值）
    
    # 棋盘比例参数
    STANDARD_BOARD_RATIO = 1.08     # 标准棋盘高宽比
    EXPAND_H_ADJUST = 0.0        # 垂直扩展的固定调整因子
    BOTTOM_ADJUST = 0.0           # 底部偏移的固定调整因子
    
    # 格子高度渐变参数
    HEIGHT_VARIATION_RATIO = 0.0 # 高度变化比例
    
    # 图像处理参数
    GAUSSIAN_BLUR_SIZE = 5
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 80
    MIN_LINE_LENGTH = 100
    MAX_LINE_GAP = 50
    
    # 角度阈值
    HORIZONTAL_ANGLE_THRESHOLD = 10  # 水平线角度阈值（度）
    VERTICAL_ANGLE_THRESHOLD = 80    # 垂直线角度阈值（度）

# ============ 用户配置 ============
USE_REGION_SELECT = True  # True=鼠标框选，False=全屏截图
SAVE_DEBUG_IMAGES = True  # 是否保存调试图像
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
        
        cv2.namedWindow("Select Region")
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


def calculate_adjustment_ratio(width, height):
    """
    根据棋盘比例计算调整参数
    对应 calculateAdjustmentRatio 函数
    """
    actual_ratio = height / width
    standard_ratio = ChessboardConfig.STANDARD_BOARD_RATIO
    
    expand_ratio_h = ChessboardConfig.EXPAND_RATIO_H
    bottom_offset_ratio = ChessboardConfig.BOTTOM_OFFSET_RATIO
    
    if actual_ratio > standard_ratio:
        expand_ratio_h += ChessboardConfig.EXPAND_H_ADJUST
        bottom_offset_ratio += ChessboardConfig.BOTTOM_ADJUST
    
    return expand_ratio_h, bottom_offset_ratio


def calculate_grid_heights(total_height):
    """
    计算渐变高度（透视效果）
    对应 calculateGridHeights 函数
    """
    rows = 10
    base_height = total_height // rows
    height_variation = base_height * ChessboardConfig.HEIGHT_VARIATION_RATIO
    grid_heights = []
    
    # 使用二次函数计算每行高度
    for i in range(rows):
        progress = i / (rows - 1)  # 0 到 1
        factor = 1 + (progress * progress * height_variation) / base_height
        grid_heights.append(int(base_height * factor))
    
    # 调整总高度以匹配原始高度
    current_total = sum(grid_heights)
    scale = total_height / current_total
    grid_heights = [int(h * scale) for h in grid_heights]
    
    return grid_heights


def segment_chessboard(cropped_region, expand_ratio_w, expand_ratio_h, bottom_offset_ratio, save_debug=False):
    """
    分割棋盘为90个格子（使用霍夫变换+KMeans）
    对应 segmentChessboard 函数
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
        return [], {}, []
    
    # 4. 分类水平线和垂直线
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        if abs(angle) < ChessboardConfig.HORIZONTAL_ANGLE_THRESHOLD:
            horizontal_lines.append([x1, y1, x2, y2])
        elif abs(angle) > ChessboardConfig.VERTICAL_ANGLE_THRESHOLD:
            vertical_lines.append([x1, y1, x2, y2])
    
    print(f"   检测到 {len(horizontal_lines)} 条水平线")
    print(f"   检测到 {len(vertical_lines)} 条垂直线")
    
    if len(horizontal_lines) < 10 or len(vertical_lines) < 9:
        print("❌ 线条不足，无法进行KMeans聚类")
        return [], {}, []
    
    # 5. KMeans聚类找到10条横线和9条竖线
    h_y_positions = []
    for line in horizontal_lines:
        h_y_positions.extend([line[1], line[3]])
    
    v_x_positions = []
    for line in vertical_lines:
        v_x_positions.extend([line[0], line[2]])
    
    # KMeans聚类
    kmeans_h = KMeans(n_clusters=10, random_state=0, n_init=10)
    kmeans_h.fit(np.array(h_y_positions).reshape(-1, 1))
    h_centers = sorted(kmeans_h.cluster_centers_.flatten())
    
    kmeans_v = KMeans(n_clusters=9, random_state=0, n_init=10)
    kmeans_v.fit(np.array(v_x_positions).reshape(-1, 1))
    v_centers = sorted(kmeans_v.cluster_centers_.flatten())
    
    print(f"   KMeans聚类完成")
    
    # 6. 确定棋盘边界
    min_y = int(h_centers[0])
    max_y = int(h_centers[-1])
    min_x = int(v_centers[0])
    max_x = int(v_centers[-1])
    
    # 7. 底部偏移调整
    board_height = max_y - min_y
    bottom_offset = int(board_height * bottom_offset_ratio)
    adjusted_max_y = max_y - bottom_offset
    
    # 8. 裁剪棋盘区域
    chessboard_region = cropped_region[min_y:adjusted_max_y, min_x:max_x]
    
    # 9. 扩展边界
    region_w = max_x - min_x
    region_h = adjusted_max_y - min_y
    expand_w = int(region_w * expand_ratio_w)
    expand_h = int(region_h * expand_ratio_h)
    
    new_x = max(0, min_x - expand_w)
    new_y = max(0, min_y - expand_h)
    new_w = min(cropped_region.shape[1] - new_x, region_w + 2 * expand_w)
    new_h = min(cropped_region.shape[0] - new_y, region_h + 2 * expand_h)
    
    expanded_region = cropped_region[new_y:new_y+new_h, new_x:new_x+new_w]
    
    if save_debug:
        cv2.imwrite("debug_03_expanded_board.png", expanded_region)
    
    # 10. 分割90个格子（渐变高度）
    rows, cols = 10, 9
    grid_width = new_w // cols
    grid_heights = calculate_grid_heights(new_h)
    
    grid_cells = []
    current_y = 0
    
    for i in range(rows):
        grid_height = grid_heights[i]
        for j in range(cols):
            x1 = j * grid_width
            x2 = (j + 1) * grid_width
            
            cell = expanded_region[current_y:current_y+grid_height, x1:x2]
            grid_cells.append(cell)
        
        current_y += grid_height
    
    expanded_rect = {'x': new_x, 'y': new_y, 'width': new_w, 'height': new_h}
    
    return grid_cells, expanded_rect, grid_heights


def detect_and_extract_chessboard(image, save_debug=False):
    """
    检测并提取棋盘（主函数）
    对应 detectAndExtractChessboard 函数
    
    Returns:
        grid_cells: 90个格子图像列表
        chessboard_rect: 棋盘位置 {'x', 'y', 'width', 'height'}
        grid_heights: 每行的高度列表
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
    
    # 7. 裁剪轮廓区域
    cropped_region = image[y:y+h, x:x+w]
    
    # 8. 计算调整参数
    expand_ratio_h, bottom_offset_ratio = calculate_adjustment_ratio(w, h)
    
    # 9. 分割棋盘
    print("\n步骤2：分割棋盘为90个格子")
    print("-"*60)
    grid_cells, expanded_rect, grid_heights = segment_chessboard(
        cropped_region,
        ChessboardConfig.EXPAND_RATIO_W,
        expand_ratio_h,
        bottom_offset_ratio,
        save_debug
    )
    
    if len(grid_cells) == 0:
        return [], {}, []
    
    # 10. 计算最终棋盘位置
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
    print("棋盘检测测试 - 复刻 xiangqi-analysis")
    print("="*60)
    
    # 获取图像
    if USE_REGION_SELECT:
        selector = RegionSelector()
        time.sleep(3)
        image = selector.select_region()
        if image is None:
            print("❌ 未选择区域")
            return
    else:
        print("📸 将在3秒后截取全屏...")
        time.sleep(3)
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
    
    # 保存一些格子样本
    if SAVE_DEBUG_IMAGES:
        print("\n💾 保存格子样本...")
        sample_positions = [(0, 0), (0, 4), (0, 8), (7, 4), (9, 0), (9, 4), (9, 8)]
        for row, col in sample_positions:
            idx = row * 9 + col
            if idx < len(grid_cells):
                filename = f"debug_cell_{row+1}-{col+1}.png"
                cv2.imwrite(filename, grid_cells[idx])
                print(f"   保存: {filename}")
    
    print("\n" + "="*60)
    print("✅ 棋盘检测完成！")
    print("="*60)
    print(f"📊 统计信息:")
    print(f"   格子数量: {len(grid_cells)}")
    print(f"   棋盘位置: ({chessboard_rect['x']}, {chessboard_rect['y']})")
    print(f"   棋盘尺寸: {chessboard_rect['width']} x {chessboard_rect['height']}")
    print(f"   格子高度: {grid_heights}")


if __name__ == "__main__":
    main()

