"""
步骤1改进版v3：基于颜色提取的棋盘检测
使用HSV颜色空间提取棋盘边框颜色，然后找到最大矩形轮廓
"""

import cv2
import numpy as np
from PIL import ImageGrab
import time

# ============ 配置参数 ============
class ChessboardConfig:
    # 棋盘边框颜色HSV范围
    # 基准颜色: HSV(29, 50%, 84%) = HSV(29, 128, 214)
    # 
    # 【调整说明】
    # - 如果提取的棋盘不完整（边框断裂）：增大H、S、V的范围
    # - 如果提取了太多背景噪声：缩小H、S、V的范围
    # 
    # H (色相): 29° ± 10° = [19, 39]
    #   - 增大范围：改为 ±15° 或 ±20°
    #   - 缩小范围：改为 ±5° 或 ±8°
    # 
    # S (饱和度): 50% ± 30% = [20%, 80%] = [51, 204]
    #   - 增大范围：改为下界30或20，上界255
    #   - 缩小范围：改为下界80或100
    # 
    # V (明度): 84% ± 20% = [64%, 100%] = [163, 255]
    #   - 增大范围：改为下界120或100
    #   - 缩小范围：改为下界180或200
    
    BOARD_HSV_LOWER = (11, 51, 163)   # 下界：H-10, S-30%, V-20%
    BOARD_HSV_UPPER = (47, 204, 255)  # 上界：H+10, S+30%, V+16%
    
    # 形态学操作参数
    MORPH_KERNEL_SIZE = 5      # 形态学核大小（增大可以连接断裂的边框）
    MORPH_CLOSE_ITERATIONS = 3  # 闭运算迭代次数（填充小孔洞）
    MORPH_OPEN_ITERATIONS = 2   # 开运算迭代次数（去除小噪声）
    
    # 轮廓筛选参数
    MIN_CONTOUR_AREA = 50000    # 最小轮廓面积（像素²）
    ASPECT_RATIO_MIN = 0.9      # 宽高比最小值（棋盘接近正方形，10:9=1.11）
    ASPECT_RATIO_MAX = 1.3      # 宽高比最大值
    
    # 棋盘比例（高/宽）
    BOARD_HEIGHT_RATIO = 10.0 / 9.0  # 10行9列，高度是宽度的10/9
    
    # 边界扩展（像素）- 用于微调棋盘边界
    # 正数表示向外扩展，负数表示向内收缩
    EXPAND_TOP = 0      # 向上扩展（负数表示向下收缩）
    EXPAND_BOTTOM = 0   # 向下扩展（负数表示向上收缩）
    EXPAND_LEFT = 0     # 向左扩展（负数表示向右收缩）
    EXPAND_RIGHT = 0    # 向右扩展（负数表示向左收缩）

# ============ 用户配置 ============
USE_REGION_SELECT = True
SAVE_DEBUG_IMAGES = True
OUTPUT_DIR = "."  # 中间结果输出到当前文件夹
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


def extract_board_color_mask(image, save_debug=False):
    """
    提取棋盘边框颜色的掩码
    
    参数:
        image: BGR格式的图像
        save_debug: 是否保存调试图像
    
    返回:
        mask: 二值掩码（棋盘边框区域=255，其他=0）
    """
    print("\n步骤1：提取棋盘边框颜色")
    print("-"*60)
    
    # 1. 转HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(f"   转换为HSV颜色空间")
    
    # 2. 创建棋盘颜色掩码
    mask = cv2.inRange(hsv_image, 
                       ChessboardConfig.BOARD_HSV_LOWER, 
                       ChessboardConfig.BOARD_HSV_UPPER)
    
    print(f"   HSV范围: H={ChessboardConfig.BOARD_HSV_LOWER[0]}-{ChessboardConfig.BOARD_HSV_UPPER[0]}°, "
          f"S={ChessboardConfig.BOARD_HSV_LOWER[1]}-{ChessboardConfig.BOARD_HSV_UPPER[1]}, "
          f"V={ChessboardConfig.BOARD_HSV_LOWER[2]}-{ChessboardConfig.BOARD_HSV_UPPER[2]}")
    
    # 统计棋盘颜色像素
    board_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    board_ratio = board_pixels / total_pixels
    
    print(f"   棋盘颜色像素: {board_pixels}/{total_pixels} ({board_ratio*100:.2f}%)")
    
    if save_debug:
        cv2.imwrite(f"{OUTPUT_DIR}/debug_01_color_mask.png", mask)
        print(f"   💾 已保存: debug_01_color_mask.png")
    
    return mask


def find_chessboard_contour(mask, save_debug=False):
    """
    从掩码中找到棋盘轮廓
    
    参数:
        mask: 二值掩码
        save_debug: 是否保存调试图像
    
    返回:
        contour_rect: 棋盘矩形 (x, y, w, h)，失败返回None
    """
    print("\n步骤2：查找棋盘轮廓")
    print("-"*60)
    
    # 1. 形态学闭运算（连接断裂的边框，填充小孔洞）
    kernel = np.ones((ChessboardConfig.MORPH_KERNEL_SIZE, 
                     ChessboardConfig.MORPH_KERNEL_SIZE), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                   iterations=ChessboardConfig.MORPH_CLOSE_ITERATIONS)
    print(f"   形态学闭运算: kernel={ChessboardConfig.MORPH_KERNEL_SIZE}x{ChessboardConfig.MORPH_KERNEL_SIZE}, "
          f"iterations={ChessboardConfig.MORPH_CLOSE_ITERATIONS}")
    
    if save_debug:
        cv2.imwrite(f"{OUTPUT_DIR}/debug_02_closed_mask.png", closed_mask)
        print(f"   💾 已保存: debug_02_closed_mask.png")
    
    # 2. 形态学开运算（去除小噪声）
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, 
                                   iterations=ChessboardConfig.MORPH_OPEN_ITERATIONS)
    print(f"   形态学开运算: iterations={ChessboardConfig.MORPH_OPEN_ITERATIONS}")
    
    if save_debug:
        cv2.imwrite(f"{OUTPUT_DIR}/debug_03_opened_mask.png", opened_mask)
        print(f"   💾 已保存: debug_03_opened_mask.png")
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   检测到 {len(contours)} 个轮廓")
    
    if len(contours) == 0:
        print("   ❌ 未检测到轮廓")
        return None
    
    # 4. 筛选有效轮廓
    valid_contours = []
    
    print(f"   筛选条件:")
    print(f"     - 最小面积: {ChessboardConfig.MIN_CONTOUR_AREA} 像素²")
    print(f"     - 宽高比: {ChessboardConfig.ASPECT_RATIO_MIN} ~ {ChessboardConfig.ASPECT_RATIO_MAX}")
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = h / w  # 高/宽，棋盘是10:9=1.11
        
        print(f"   轮廓{i+1}: 位置=({x},{y}), 尺寸={w}x{h}, "
              f"面积={area:.0f}, 宽高比={aspect_ratio:.2f}")
        
        # 条件1：面积足够大
        if area >= ChessboardConfig.MIN_CONTOUR_AREA:
            # 条件2：宽高比接近正方形（10:9）
            if ChessboardConfig.ASPECT_RATIO_MIN <= aspect_ratio <= ChessboardConfig.ASPECT_RATIO_MAX:
                valid_contours.append({
                    'contour': contour,
                    'rect': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
                print(f"     ✓ 有效轮廓")
            else:
                print(f"     ✗ 宽高比不符")
        else:
            print(f"     ✗ 面积不足")
    
    # 5. 选择最大的轮廓
    if len(valid_contours) == 0:
        print("   ❌ 没有有效轮廓")
        return None
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    print(f"\n   选择最大轮廓:")
    print(f"     位置: ({x}, {y})")
    print(f"     尺寸: {w}x{h}")
    print(f"     面积: {best_contour['area']:.0f}")
    print(f"     宽高比: {best_contour['aspect_ratio']:.3f}")
    
    return (x, y, w, h)


def segment_chessboard(image, board_rect, save_debug=False):
    """
    分割棋盘为90个格子
    
    参数:
        image: 原始图像
        board_rect: 棋盘矩形 (x, y, w, h)
        save_debug: 是否保存调试图像
    
    返回:
        grid_cells: 90个格子图像列表
        final_rect: 最终棋盘位置
        grid_heights: 格子高度列表
    """
    print("\n步骤3：分割棋盘")
    print("-"*60)
    
    x, y, w, h = board_rect
    
    print(f"   原始棋盘: 位置=({x}, {y}), 尺寸={w}x{h}")
    
    # 1. 应用边界扩展
    expand_top = ChessboardConfig.EXPAND_TOP
    expand_bottom = ChessboardConfig.EXPAND_BOTTOM
    expand_left = ChessboardConfig.EXPAND_LEFT
    expand_right = ChessboardConfig.EXPAND_RIGHT
    
    print(f"   边界扩展: 上={expand_top}px, 下={expand_bottom}px, 左={expand_left}px, 右={expand_right}px")
    
    # 计算扩展后的区域
    expanded_x = x - expand_left
    expanded_y = y - expand_top
    expanded_w = w + expand_left + expand_right
    expanded_h = h + expand_top + expand_bottom
    
    # 边界检查
    expanded_x = max(0, expanded_x)
    expanded_y = max(0, expanded_y)
    
    if expanded_y + expanded_h > image.shape[0]:
        expanded_h = image.shape[0] - expanded_y
        print(f"   ⚠️  高度超出边界，调整为: {expanded_h}")
    
    if expanded_x + expanded_w > image.shape[1]:
        expanded_w = image.shape[1] - expanded_x
        print(f"   ⚠️  宽度超出边界，调整为: {expanded_w}")
    
    print(f"   扩展后棋盘: 位置=({expanded_x}, {expanded_y}), 尺寸={expanded_w}x{expanded_h}")
    
    # 2. 裁剪棋盘区域
    board_image = image[expanded_y:expanded_y+expanded_h, 
                       expanded_x:expanded_x+expanded_w]
    
    if save_debug:
        cv2.imwrite(f"{OUTPUT_DIR}/debug_04_board.png", board_image)
        print(f"   💾 已保存: debug_04_board.png")
    
    # 3. 均匀分割 10行 x 9列
    rows, cols = 10, 9
    cell_height = expanded_h // rows
    cell_width = expanded_w // cols
    
    print(f"   格子尺寸: {cell_width} x {cell_height}")
    
    grid_cells = []
    
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height if i < rows - 1 else expanded_h
            x1 = j * cell_width
            x2 = (j + 1) * cell_width if j < cols - 1 else expanded_w
            
            cell = board_image[y1:y2, x1:x2]
            grid_cells.append(cell)
    
    final_rect = {
        'x': expanded_x,
        'y': expanded_y,
        'width': expanded_w,
        'height': expanded_h
    }
    
    grid_heights = [cell_height] * rows
    
    return grid_cells, final_rect, grid_heights


def detect_and_extract_chessboard(image, save_debug=False):
    """
    检测并提取棋盘（主函数）- 基于颜色提取
    """
    print("\n" + "="*60)
    print("基于颜色提取的棋盘检测")
    print("="*60)
    
    # 步骤1：提取棋盘边框颜色掩码
    mask = extract_board_color_mask(image, save_debug)
    
    # 步骤2：查找棋盘轮廓
    board_rect = find_chessboard_contour(mask, save_debug)
    
    if board_rect is None:
        print("\n❌ 未找到棋盘轮廓")
        return [], {}, []
    
    # 步骤3：分割棋盘为90个格子
    grid_cells, final_rect, grid_heights = segment_chessboard(image, board_rect, save_debug)
    
    print(f"\n✓ 分割完成: {len(grid_cells)} 个格子")
    print(f"✓ 最终棋盘位置: {final_rect}")
    
    return grid_cells, final_rect, grid_heights


def main():
    """主函数"""
    print("="*60)
    print("棋盘检测测试 - 基于颜色提取")
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
        cv2.imwrite(f"{OUTPUT_DIR}/debug_00_original.png", image)
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
    
    print(f"\n📁 中间结果文件:")
    print(f"   {OUTPUT_DIR}/debug_00_original.png      - 原始图像")
    print(f"   {OUTPUT_DIR}/debug_01_color_mask.png    - 颜色掩码")
    print(f"   {OUTPUT_DIR}/debug_02_closed_mask.png   - 闭运算后")
    print(f"   {OUTPUT_DIR}/debug_03_opened_mask.png   - 开运算后")
    print(f"   {OUTPUT_DIR}/debug_04_board.png         - 最终棋盘")
    print(f"   step1_output/                           - 90个格子")


if __name__ == "__main__":
    main()

