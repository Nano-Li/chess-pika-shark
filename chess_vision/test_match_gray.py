"""
模板匹配测试工具 - 灰度匹配版本
模板和格子都转为灰度后再匹配，只比较明暗分布
"""

import cv2
import numpy as np
from PIL import ImageGrab
import time

# ============ 配置区域 ============
TEMPLATE_PATH = r"D:\Projects\AIchess\chess_vision\output1_useful\1-5.png"  # 测试用的模板路径
MATCH_THRESHOLD = 0.5              # 匹配阈值（0-1）
USE_REGION_SELECT = True           # 是否使用区域选择（False则全屏截图）
# =================================


class RegionSelector:
    """鼠标框选区域工具"""
    
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.screenshot = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
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
        """选择截图区域"""
        print("📸 正在截取全屏...")
        screenshot = ImageGrab.grab()
        self.screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # 缩小显示（方便在屏幕上显示）
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
            
            # 绘制选择框
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
        
        # 转换回原始坐标
        x1 = int(min(self.start_point[0], self.end_point[0]) / scale)
        y1 = int(min(self.start_point[1], self.end_point[1]) / scale)
        x2 = int(max(self.start_point[0], self.end_point[0]) / scale)
        y2 = int(max(self.start_point[1], self.end_point[1]) / scale)
        
        # 裁剪区域
        region = self.screenshot[y1:y2, x1:x2]
        
        print(f"✅ 已选择区域: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"📐 区域尺寸: {x2-x1} x {y2-y1}")
        
        return region


def detect_chessboard(image):
    """检测棋盘边框（复用v2）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("未检测到棋盘轮廓")
    
    valid_contours = []
    img_area = image.shape[0] * image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.1:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        if 0.7 < aspect_ratio < 1.3:
            valid_contours.append((contour, area, (x, y, w, h)))
    
    if not valid_contours:
        raise Exception("未找到符合条件的棋盘")
    
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    _, _, (x, y, w, h) = valid_contours[0]
    
    board_image = image[y:y+h, x:x+w]
    
    return board_image, (x, y, w, h)


def segment_board(board_image):
    """分割90个格子（复用v2）"""
    h, w = board_image.shape[:2]
    rows, cols = 10, 9
    cell_height = h // rows
    cell_width = w // cols
    
    cells = []
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            
            cell = board_image[y1:y2, x1:x2]
            cells.append((cell, (i+1, j+1)))  # 从1开始
    
    return cells


def match_template_gray(cell, template):
    """灰度模板匹配（转为单通道）"""
    # 调整模板大小到格子大小
    cell_h, cell_w = cell.shape[:2]
    template_resized = cv2.resize(template, (cell_w, cell_h))
    
    # 转灰度（去除颜色信息）
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    
    # 灰度图像匹配（只比较明暗）
    result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val


def test_template_matching(template_path, threshold, use_region_select):
    """测试模板匹配"""
    
    print("=" * 60)
    print("模板匹配测试工具 - 灰度匹配版本")
    print("=" * 60)
    print()
    
    # 加载模板
    print(f"📂 加载模板: {template_path}")
    template = cv2.imread(template_path)
    if template is None:
        print(f"❌ 无法读取模板: {template_path}")
        return
    
    print(f"📏 模板尺寸: {template.shape[1]} x {template.shape[0]}")
    print(f"⚫ 匹配模式: 灰度匹配（单通道，忽略颜色）")
    print(f"🎯 匹配阈值: {threshold}")
    print()
    
    # 获取图像
    if use_region_select:
        selector = RegionSelector()
        image = selector.select_region()
        if image is None:
            print("❌ 未选择区域")
            return
    else:
        print("📸 将在3秒后截取全屏...")
        time.sleep(3)
        screenshot = ImageGrab.grab()
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    print()
    print("🔍 步骤1: 检测棋盘...")
    board_image, board_rect = detect_chessboard(image)
    print(f"   ✓ 棋盘位置: {board_rect}")
    
    print("✂️  步骤2: 分割格子...")
    cells = segment_board(board_image)
    print(f"   ✓ 分割完成: {len(cells)} 个格子")
    
    print(f"🎯 步骤3: 灰度模板匹配（阈值 > {threshold}）...")
    print()
    
    matches = []
    for cell, (row, col) in cells:
        score = match_template_gray(cell, template)
        if score > threshold:
            matches.append((row, col, score))
    
    # 输出结果
    print("=" * 60)
    print(f"✅ 匹配完成！找到 {len(matches)} 个匹配位置")
    print("=" * 60)
    print()
    
    if matches:
        print("匹配位置（按相似度排序）：")
        matches.sort(key=lambda x: x[2], reverse=True)
        for row, col, score in matches:
            print(f"  位置 {row}-{col}: 相似度 {score:.2%}")
    else:
        print("未找到匹配位置")
        print()
        print("建议：")
        print("  1. 降低阈值（当前: {:.0%}）".format(threshold))
        print("  2. 检查模板是否正确")
        print("  3. 检查截图是否包含目标")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    test_template_matching(TEMPLATE_PATH, MATCH_THRESHOLD, USE_REGION_SELECT)

