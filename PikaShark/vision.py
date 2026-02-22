"""
视觉模块：截图、棋盘识别、FEN码生成
"""

import cv2
import numpy as np
from PIL import ImageGrab
import os


# ============ 配置参数 ============
class VisionConfig:
    # 棋盘边框颜色HSV范围（用于颜色提取检测）
    # 基准颜色: HSV(29, 50%, 84%) = HSV(29, 128, 214)
    # 
    # 【调整说明】
    # - 如果提取的棋盘不完整（边框断裂）：增大H、S、V的范围
    # - 如果提取了太多背景噪声：缩小H、S、V的范围
    BOARD_HSV_LOWER = (11, 51, 163)   # 下界：H-10, S-30%, V-20%
    BOARD_HSV_UPPER = (47, 204, 255)  # 上界：H+10, S+30%, V+16%
    
    # 形态学操作参数（棋盘检测）
    BOARD_MORPH_KERNEL_SIZE = 5      # 形态学核大小
    BOARD_MORPH_CLOSE_ITERATIONS = 3  # 闭运算迭代次数（连接断裂边框）
    BOARD_MORPH_OPEN_ITERATIONS = 2   # 开运算迭代次数（去除噪声）
    
    # 轮廓筛选参数（棋盘检测）
    MIN_CONTOUR_AREA = 50000    # 最小轮廓面积（像素²）
    ASPECT_RATIO_MIN = 0.9      # 宽高比最小值（棋盘接近正方形，10:9=1.11）
    ASPECT_RATIO_MAX = 1.3      # 宽高比最大值
    
    # 棋盘比例
    BOARD_HEIGHT_RATIO = 10.0 / 9.0
    
    # 边界微调（像素）
    EXPAND_TOP = 0
    EXPAND_BOTTOM = 0
    EXPAND_LEFT = 0
    EXPAND_RIGHT = 0
    
    # 模板匹配参数
    MATCHING_METHOD = cv2.TM_CCOEFF_NORMED
    MATCH_THRESHOLD = 0.3  # 参考项目使用的阈值
    
    # 棋子检测参数
    CONTRAST_THRESHOLD = 30  # 对比度阈值（标准差）
    
    # 颜色像素比例阈值（用于检测棋子颜色）
    RED_RATIO_THRESHOLD = 0.05
    BLACK_RATIO_THRESHOLD = 0.03
    
    # 红色HSV范围（用于检测棋子颜色）
    RED_HSV_LOWER_1 = (0, 120, 120)
    RED_HSV_UPPER_1 = (10, 255, 255)
    RED_HSV_LOWER_2 = (160, 120, 120)
    RED_HSV_UPPER_2 = (179, 255, 255)
    
    # 黑色HSV范围（用于检测棋子颜色）
    BLACK_HSV_LOWER = (0, 0, 0)
    BLACK_HSV_UPPER = (180, 255, 80)
    
    # 红色棋子预处理参数（用于模板匹配）
    RED_PREPROCESS_HSV_LOWER_1 = (0, 100, 100)
    RED_PREPROCESS_HSV_UPPER_1 = (10, 255, 255)
    RED_PREPROCESS_HSV_LOWER_2 = (160, 100, 100)
    RED_PREPROCESS_HSV_UPPER_2 = (179, 255, 255)
    
    # 红色轮廓筛选参数
    RED_ASPECT_RATIO_MIN = 0.8
    RED_ASPECT_RATIO_MAX = 1.2
    RED_AREA_RATIO_MIN = 0.3
    
    # 黑色棋子预处理参数（用于模板匹配）
    BLACK_PREPROCESS_HSV_LOWER = (0, 0, 0)
    BLACK_PREPROCESS_HSV_UPPER = (180, 255, 80)
    
    # Canny边缘检测参数（黑色棋子）
    BLACK_CANNY_THRESHOLD1 = 400
    BLACK_CANNY_THRESHOLD2 = 600
    
    # 轮廓缩放因子（黑色棋子）
    BLACK_CONTOUR_SCALE_FACTOR = 1.00
    
    # 黑色轮廓筛选参数
    BLACK_ASPECT_RATIO_MIN = 0.3
    BLACK_ASPECT_RATIO_MAX = 1.8
    BLACK_AREA_RATIO_MIN = 0.1
    
    # 形态学操作参数（红色）
    RED_MORPH_KERNEL_SIZE = 3
    RED_MORPH_ITERATIONS = 3

    # 形态学操作参数（黑色）
    BLACK_MORPH_KERNEL_SIZE = 3
    BLACK_MORPH_ITERATIONS = 4

    # 噪点过滤阈值（相对于总面积的比例）
    NOISE_THRESHOLD = 0.01  # 1%
    
    # 调试输出
    DEBUG_ENABLED = True
    DEBUG_DIR = "./debug"


# ============ 全局变量 ============
class VisionState:
    """视觉模块状态"""
    def __init__(self):
        self.bbox = None  # 棋盘边界框 (x1, y1, x2, y2)
        self.templates = {}  # 模板字典 {piece_name: gray_template}
        self.initialized = False


vision_state = VisionState()


# ============ 区域框选 ============
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
        """框选区域并返回bbox"""
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
        cv2.setWindowProperty("Select Region", cv2.WND_PROP_TOPMOST, 1)
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
        
        print(f"✅ 已选择区域: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"📐 区域尺寸: {x2-x1} x {y2-y1}")
        
        return (x1, y1, x2, y2)


# ============ 棋盘检测 ============
def extract_board_color_mask(image):
    """
    提取棋盘边框颜色的掩码（基于HSV颜色提取）
    
    参数:
        image: BGR格式的图像
    
    返回:
        mask: 二值掩码（棋盘边框区域=255，其他=0）
    """
    # 转HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建棋盘颜色掩码
    mask = cv2.inRange(hsv_image, 
                       VisionConfig.BOARD_HSV_LOWER, 
                       VisionConfig.BOARD_HSV_UPPER)
    
    # 统计棋盘颜色像素
    board_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    board_ratio = board_pixels / total_pixels
    
    print(f"   棋盘颜色像素: {board_pixels}/{total_pixels} ({board_ratio*100:.2f}%)")
    
    return mask


def find_chessboard_contour(mask):
    """
    从掩码中找到棋盘轮廓
    
    参数:
        mask: 二值掩码
    
    返回:
        contour_rect: 棋盘矩形 (x, y, w, h)，失败返回None
    """
    # 1. 形态学闭运算（连接断裂的边框，填充小孔洞）
    kernel = np.ones((VisionConfig.BOARD_MORPH_KERNEL_SIZE, 
                     VisionConfig.BOARD_MORPH_KERNEL_SIZE), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                   iterations=VisionConfig.BOARD_MORPH_CLOSE_ITERATIONS)
    
    # 2. 形态学开运算（去除小噪声）
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, 
                                   iterations=VisionConfig.BOARD_MORPH_OPEN_ITERATIONS)
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   检测到 {len(contours)} 个轮廓")
    
    if len(contours) == 0:
        return None
    
    # 4. 筛选有效轮廓
    valid_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = h / w  # 高/宽，棋盘是10:9=1.11
        
        # 条件1：面积足够大
        if area >= VisionConfig.MIN_CONTOUR_AREA:
            # 条件2：宽高比接近正方形（10:9）
            if VisionConfig.ASPECT_RATIO_MIN <= aspect_ratio <= VisionConfig.ASPECT_RATIO_MAX:
                valid_contours.append({
                    'contour': contour,
                    'rect': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
    
    # 5. 选择最大的轮廓
    if len(valid_contours) == 0:
        print("   ❌ 没有有效轮廓")
        return None
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    print(f"   选择最大轮廓: 位置=({x}, {y}), 尺寸={w}x{h}, 宽高比={best_contour['aspect_ratio']:.3f}")
    
    return (x, y, w, h)


def split_chessboard_to_cells(image, debug_prefix=""):
    """
    将棋盘图像分割为90个格子（基于颜色提取）
    返回: list of 90 cell images
    """
    # 创建调试目录
    if VisionConfig.DEBUG_ENABLED:
        os.makedirs(VisionConfig.DEBUG_DIR, exist_ok=True)
    
    print("   使用颜色提取方法检测棋盘...")
    
    # 1. 提取棋盘边框颜色掩码
    mask = extract_board_color_mask(image)
    
    if VisionConfig.DEBUG_ENABLED:
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, f"{debug_prefix}02_color_mask.png"), mask)
        print(f"   💾 已保存: {debug_prefix}02_color_mask.png")
    
    # 2. 查找棋盘轮廓
    board_rect = find_chessboard_contour(mask)
    
    if board_rect is None:
        print("❌ 未找到棋盘轮廓")
        return []
    
    x, y, w, h = board_rect
    
    print(f"   棋盘位置: ({x}, {y})")
    print(f"   棋盘尺寸: {w}x{h}")
    
    # 调试：保存检测到的棋盘区域
    if VisionConfig.DEBUG_ENABLED:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, f"{debug_prefix}02_detected_board.png"), debug_img)
        print(f"   💾 已保存: {debug_prefix}02_detected_board.png")
    
    # 3. 应用边界微调
    expanded_x = x - VisionConfig.EXPAND_LEFT
    expanded_y = y - VisionConfig.EXPAND_TOP
    expanded_w = w + VisionConfig.EXPAND_LEFT + VisionConfig.EXPAND_RIGHT
    expanded_h = h + VisionConfig.EXPAND_TOP + VisionConfig.EXPAND_BOTTOM
    
    # 边界检查
    expanded_x = max(0, expanded_x)
    expanded_y = max(0, expanded_y)
    
    if expanded_y + expanded_h > image.shape[0]:
        expanded_h = image.shape[0] - expanded_y
    
    if expanded_x + expanded_w > image.shape[1]:
        expanded_w = image.shape[1] - expanded_x
    
    print(f"   边界微调后: ({expanded_x}, {expanded_y}), {expanded_w}x{expanded_h}")
    
    # 4. 裁剪棋盘
    chessboard = image[expanded_y:expanded_y+expanded_h, 
                      expanded_x:expanded_x+expanded_w]
    
    # 调试：保存裁剪后的棋盘
    if VisionConfig.DEBUG_ENABLED:
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, f"{debug_prefix}03_cropped_board.png"), chessboard)
        print(f"   💾 已保存: {debug_prefix}03_cropped_board.png")
    
    # 5. 均匀分割为90个格子
    rows, cols = 10, 9
    cell_height = expanded_h // rows
    cell_width = expanded_w // cols
    
    print(f"   格子尺寸: {cell_width}x{cell_height}")
    
    # 调试：保存带网格线的棋盘
    if VisionConfig.DEBUG_ENABLED:
        debug_grid = chessboard.copy()
        # 绘制垂直线
        for col in range(cols + 1):
            x_line = int(col * cell_width)
            cv2.line(debug_grid, (x_line, 0), (x_line, expanded_h), (0, 255, 0), 1)
        # 绘制水平线
        for row in range(rows + 1):
            y_line = int(row * cell_height)
            cv2.line(debug_grid, (0, y_line), (expanded_w, y_line), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, f"{debug_prefix}04_grid.png"), debug_grid)
        print(f"   💾 已保存: {debug_prefix}04_grid.png")
    
    cells = []
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height if i < rows - 1 else expanded_h
            x1 = j * cell_width
            x2 = (j + 1) * cell_width if j < cols - 1 else expanded_w
            
            cell = chessboard[y1:y2, x1:x2]
            cells.append(cell)
    
    return cells


# ============ 模板加载 ============
def load_templates():
    """
    加载14个模板图像（红7个 + 黑7个）
    返回: {piece_name: gray_template}
    """
    templates = {}
    
    # 模板命名映射（按照PIECE_TYPE_MAP的英文命名）
    red_pieces = ['king', 'guard', 'bishop', 'knight', 'rook', 'cannon', 'pawn']
    black_pieces = ['king', 'guard', 'bishop', 'knight', 'rook', 'cannon', 'pawn']
    
    # 加载红色棋子模板
    for piece in red_pieces:
        path = f"./templates/red/{piece}.png"
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 转灰度
                if len(img.shape) == 2:
                    gray = img
                elif img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                templates[f'r_{piece}'] = gray
                print(f"   ✓ 加载模板: r_{piece} ({path})")
            else:
                print(f"   ⚠️  无法读取: {path}")
        else:
            print(f"   ⚠️  文件不存在: {path}")
    
    # 加载黑色棋子模板
    for piece in black_pieces:
        path = f"./templates/black/{piece}.png"
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 转灰度
                if len(img.shape) == 2:
                    gray = img
                elif img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                templates[f'b_{piece}'] = gray
                print(f"   ✓ 加载模板: b_{piece} ({path})")
            else:
                print(f"   ⚠️  无法读取: {path}")
        else:
            print(f"   ⚠️  文件不存在: {path}")
    
    return templates


# ============ 棋子识别 ============
def is_empty_cell(cell_image):
    """
    判断格子是否为空（基于对比度 - 标准差）
    
    参数:
        cell_image: 格子图像（BGR）
    
    返回:
        is_empty: 是否为空
        contrast: 对比度值（标准差）
    """
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # 使用标准差作为对比度（更稳定）
    mean, stddev = cv2.meanStdDev(gray)
    contrast = stddev[0][0]
    
    is_empty = contrast < VisionConfig.CONTRAST_THRESHOLD
    
    return is_empty


def detect_piece_color(cell_image):
    """
    判断棋子颜色（红/黑/无）
    
    参数:
        cell_image: 格子图像（BGR）
    
    返回:
        color: 'red', 'black', 或 'none'
        red_ratio: 红色像素比例
        black_ratio: 黑色像素比例
    """
    # 转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    
    # 创建红色掩码（两个范围）
    red_mask1 = cv2.inRange(hsv_image, 
                           VisionConfig.RED_HSV_LOWER_1,
                           VisionConfig.RED_HSV_UPPER_1)
    red_mask2 = cv2.inRange(hsv_image,
                           VisionConfig.RED_HSV_LOWER_2,
                           VisionConfig.RED_HSV_UPPER_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # 创建黑色掩码
    black_mask = cv2.inRange(hsv_image,
                            VisionConfig.BLACK_HSV_LOWER,
                            VisionConfig.BLACK_HSV_UPPER)
    
    # 计算像素比例
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    red_pixels = cv2.countNonZero(red_mask)
    black_pixels = cv2.countNonZero(black_mask)
    
    red_ratio = red_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    
    # 判断逻辑
    # 红色棋子：红色比例 >= 0.05 且 黑色比例 < 0.03
    if red_ratio >= VisionConfig.RED_RATIO_THRESHOLD and \
       black_ratio < VisionConfig.BLACK_RATIO_THRESHOLD:
        return 'red', red_ratio, black_ratio
    
    # 黑色棋子：红色比例 < 0.05 且 黑色比例 >= 0.03
    elif red_ratio < VisionConfig.RED_RATIO_THRESHOLD and \
         black_ratio >= VisionConfig.BLACK_RATIO_THRESHOLD:
        return 'black', red_ratio, black_ratio
    
    # 其他情况：没有棋子（楚河汉界等）
    else:
        return 'none', red_ratio, black_ratio


def preprocess_red_piece(cell_image):
    """
    预处理红色棋子图像（用于模板匹配）
    
    步骤：
    1. 提取红色掩码（HSV）
    2. 形态学闭运算（填充孔洞）
    3. 提取最大轮廓区域
    4. 裁剪出棋子区域
    
    参数:
        cell_image: BGR格式的格子图像
    
    返回:
        processed_image: 处理后的灰度图像
    """
    # 1. 转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    
    # 2. 创建红色掩码（两个范围）
    mask1 = cv2.inRange(hsv_image, 
                       VisionConfig.RED_PREPROCESS_HSV_LOWER_1, 
                       VisionConfig.RED_PREPROCESS_HSV_UPPER_1)
    mask2 = cv2.inRange(hsv_image, 
                       VisionConfig.RED_PREPROCESS_HSV_LOWER_2, 
                       VisionConfig.RED_PREPROCESS_HSV_UPPER_2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    
    # 3. 形态学闭运算（填充小孔洞）
    kernel = np.ones((VisionConfig.RED_MORPH_KERNEL_SIZE,
                     VisionConfig.RED_MORPH_KERNEL_SIZE), np.uint8)
    morphed_mask = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel,
                                    iterations=VisionConfig.RED_MORPH_ITERATIONS)
    
    # 4. 查找轮廓
    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # 没有轮廓，返回原始掩码
        return mask_red
    
    # 5. 筛选有效轮廓
    img_area = cell_image.shape[0] * cell_image.shape[1]
    valid_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        contour_area = w * h
        area_ratio = contour_area / img_area
        
        # 条件1：接近正方形
        if VisionConfig.RED_ASPECT_RATIO_MIN <= aspect_ratio <= VisionConfig.RED_ASPECT_RATIO_MAX:
            # 条件2：占格子面积 >= 30%
            if area_ratio >= VisionConfig.RED_AREA_RATIO_MIN:
                valid_contours.append({
                    'contour': contour,
                    'area': contour_area,
                    'rect': (x, y, w, h)
                })
    
    # 6. 选择最大的轮廓
    if len(valid_contours) == 0:
        # 没有有效轮廓，返回原始掩码
        return mask_red
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    # 7. 裁剪出轮廓区域
    cropped_image = mask_red[y:y+h, x:x+w]
    
    return cropped_image


def filter_noise_contours_in_mask(mask_image, noise_threshold=0.01):
    """
    智能过滤掩码中的噪点轮廓
    
    参数:
        mask_image: 输入掩码图像（二值图）
        noise_threshold: 噪点过滤阈值（相对于总面积的比例，默认2%）
    
    返回:
        cleaned_mask: 去除噪点后的掩码
    """
    # 1. 提取所有轮廓
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask_image
    
    # 2. 计算总面积
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    if total_area == 0:
        return mask_image
    
    min_area = total_area * noise_threshold
    
    # 3. 分类轮廓：有效轮廓 vs 噪点轮廓
    valid_contours = []
    noise_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)
        else:
            noise_contours.append(cnt)
    
    # 4. 创建清理后的掩码：保留原始掩码，只擦除噪点区域
    cleaned_mask = mask_image.copy()
    
    # 将噪点轮廓区域填充为黑色（0）
    if len(noise_contours) > 0:
        cv2.drawContours(cleaned_mask, noise_contours, -1, 0, cv2.FILLED)
    
    return cleaned_mask


def preprocess_black_piece(cell_image):
    """
    预处理黑色棋子图像（用于模板匹配）v3 - 增加智能噪点过滤
    
    步骤：
    1. HSV转换 + 直方图均衡化
    2. Canny边缘检测（高阈值）
    3. 形态学闭运算
    4. 查找最大轮廓并缩放
    5. 创建黑色掩码
    6. 两个掩码相交
    7. 智能过滤噪点（新增）
    8. 提取最大轮廓区域并裁剪
    
    参数:
        cell_image: BGR格式的格子图像
    
    返回:
        processed_image: 处理后的灰度图像
    """
    # 步骤1：转HSV颜色空间
    hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    
    # 步骤2：直方图均衡化（增强对比度）
    h, s, v = cv2.split(hsv_image)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    
    # 步骤3：Canny边缘检测（高阈值）
    edges = cv2.Canny(hsv_equalized, 
                     VisionConfig.BLACK_CANNY_THRESHOLD1,
                     VisionConfig.BLACK_CANNY_THRESHOLD2)
    
    # 步骤4：形态学闭运算（连接边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 步骤5：查找轮廓
    contours, _ = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # 没有轮廓，返回空掩码
        return np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    
    # 步骤6：找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 步骤7：计算轮廓中心
    M = cv2.moments(max_contour)
    if M['m00'] == 0:
        return np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    
    center_x = M['m10'] / M['m00']
    center_y = M['m01'] / M['m00']
    
    # 步骤8：轮廓缩放（避免包含背景）
    scaled_contour = []
    for point in max_contour:
        x, y = point[0]
        new_x = center_x + (x - center_x) * VisionConfig.BLACK_CONTOUR_SCALE_FACTOR
        new_y = center_y + (y - center_y) * VisionConfig.BLACK_CONTOUR_SCALE_FACTOR
        scaled_contour.append([[int(new_x), int(new_y)]])
    
    scaled_contour = np.array(scaled_contour, dtype=np.int32)
    
    # 步骤9：创建黑色掩码（使用原始HSV，不用均衡化的）
    hsv_original = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_original,
                            VisionConfig.BLACK_PREPROCESS_HSV_LOWER,
                            VisionConfig.BLACK_PREPROCESS_HSV_UPPER)
    
    # 步骤10：创建轮廓掩码
    contour_mask = np.zeros((cell_image.shape[0], cell_image.shape[1]), dtype=np.uint8)
    cv2.drawContours(contour_mask, [scaled_contour], 0, 255, cv2.FILLED)
    
    # 步骤11：两个掩码相交（黑色掩码 AND 轮廓掩码）
    final_mask = cv2.bitwise_and(black_mask, contour_mask)
    
    # 步骤12：智能过滤噪点（新增）
    cleaned_mask = filter_noise_contours_in_mask(final_mask, noise_threshold=VisionConfig.NOISE_THRESHOLD)
    
    # 步骤13：形态学闭运算（填充小孔洞）
    kernel = np.ones((VisionConfig.BLACK_MORPH_KERNEL_SIZE,
                     VisionConfig.BLACK_MORPH_KERNEL_SIZE), np.uint8)
    morphed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=VisionConfig.BLACK_MORPH_ITERATIONS)
    
    # 步骤14：查找轮廓（用于裁剪）
    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return cleaned_mask
    
    # 步骤15：筛选有效轮廓
    img_area = cell_image.shape[0] * cell_image.shape[1]
    valid_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        contour_area = w * h
        area_ratio = contour_area / img_area
        
        # 条件1：接近正方形（黑色棋子范围更宽）
        if VisionConfig.BLACK_ASPECT_RATIO_MIN <= aspect_ratio <= VisionConfig.BLACK_ASPECT_RATIO_MAX:
            # 条件2：占格子面积 >= 10%
            if area_ratio >= VisionConfig.BLACK_AREA_RATIO_MIN:
                valid_contours.append({
                    'contour': contour,
                    'area': contour_area,
                    'rect': (x, y, w, h)
                })
    
    # 步骤16：选择最大的轮廓
    if len(valid_contours) == 0:
        return cleaned_mask
    
    best_contour = max(valid_contours, key=lambda c: c['area'])
    x, y, w, h = best_contour['rect']
    
    # 步骤17：裁剪出轮廓区域
    cropped_image = cleaned_mask[y:y+h, x:x+w]
    
    return cropped_image


def match_piece_with_templates(cell_image, templates, piece_color):
    """
    将格子与对应颜色的模板匹配，找到最佳匹配
    
    参数:
        cell_image: 格子图像（BGR）
        templates: 模板字典
        piece_color: 'red' 或 'black'
    
    返回:
        (piece_name, match_score) 或 (None, 0)
    """
    # 1. 根据颜色预处理格子图像
    if piece_color == 'red':
        cell_processed = preprocess_red_piece(cell_image)
    else:
        cell_processed = preprocess_black_piece(cell_image)
    
    # 2. 选择对应颜色的模板前缀
    prefix = 'r_' if piece_color == 'red' else 'b_'
    
    best_match = None
    best_score = -1
    
    # 3. 遍历对应颜色的所有模板
    for piece_name, template in templates.items():
        # 只匹配对应颜色的模板
        if not piece_name.startswith(prefix):
            continue
        
        # 4. 调整模板大小到预处理后的格子大小
        cell_size = (cell_processed.shape[1], cell_processed.shape[0])
        resized_template = cv2.resize(template, cell_size)
        
        # 5. 模板匹配
        result = cv2.matchTemplate(cell_processed, resized_template, 
                                  VisionConfig.MATCHING_METHOD)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = piece_name
    
    # 6. 检查是否超过阈值
    if best_score < VisionConfig.MATCH_THRESHOLD:
        return None, best_score
    
    return best_match, best_score


def match_piece_with_templates_debug(cell_image, templates, piece_color, row, col):
    """
    将格子与对应颜色的模板匹配，找到最佳匹配（带调试输出）
    
    参数:
        cell_image: 格子图像（BGR）
        templates: 模板字典
        piece_color: 'red' 或 'black'
        row: 行号
        col: 列号
    
    返回:
        (piece_name, match_score) 或 (None, 0)
    """
    # 1. 根据颜色预处理格子图像
    if piece_color == 'red':
        cell_processed = preprocess_red_piece(cell_image)
    else:
        cell_processed = preprocess_black_piece(cell_image)
    
    # 2. 选择对应颜色的模板前缀
    prefix = 'r_' if piece_color == 'red' else 'b_'
    
    best_match = None
    best_score = -1
    
    # 3. 遍历对应颜色的所有模板
    for piece_name, template in templates.items():
        # 只匹配对应颜色的模板
        if not piece_name.startswith(prefix):
            continue
        
        # 4. 调整模板大小到预处理后的格子大小
        cell_size = (cell_processed.shape[1], cell_processed.shape[0])
        resized_template = cv2.resize(template, cell_size)
        
        # 5. 模板匹配
        result = cv2.matchTemplate(cell_processed, resized_template, 
                                  VisionConfig.MATCHING_METHOD)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = piece_name
    
    # 6. 保存黑色棋子的预处理图像（用于调试）
    if VisionConfig.DEBUG_ENABLED and piece_color == 'black':
        debug_dir = os.path.join(VisionConfig.DEBUG_DIR, "black_pieces_processed")
        os.makedirs(debug_dir, exist_ok=True)
        
        status = "matched" if best_score >= VisionConfig.MATCH_THRESHOLD else "unmatched"
        filename = f"cell_{row:02d}-{col:02d}_{status}_score{best_score:.3f}.png"
        cv2.imwrite(os.path.join(debug_dir, filename), cell_processed)
    
    # 7. 检查是否超过阈值
    if best_score < VisionConfig.MATCH_THRESHOLD:
        return None, best_score
    
    return best_match, best_score


def recognize_all_pieces(cells, templates):
    """
    识别所有格子中的棋子
    
    流程:
    1. 先检测所有格子是否有棋子及颜色（对比度+HSV）
    2. 对有棋子的格子进行模板匹配
    
    参数:
        cells: 90个格子图像列表
        templates: 模板字典
    
    返回:
        piece_positions: 字典 {(row, col): piece_name}
        detection_report: 检测报告列表
    """
    print("\n" + "="*60)
    print("识别所有棋子")
    print("="*60)
    
    piece_positions = {}
    detection_report = []
    
    # 统计信息
    total_pieces = 0
    red_pieces = 0
    black_pieces = 0
    matched_pieces = 0
    unmatched_pieces = 0
    
    print("\n步骤1：检测棋子位置和颜色")
    print("-"*60)
    
    # 第一遍：检测所有格子是否有棋子及颜色
    piece_detections = []
    for idx, cell in enumerate(cells):
        row = idx // 9 + 1
        col = idx % 9 + 1
        
        # 检测是否为空
        is_empty = is_empty_cell(cell)
        
        if not is_empty:
            # 检测颜色
            color, red_ratio, black_ratio = detect_piece_color(cell)
            
            if color in ['red', 'black']:
                piece_detections.append({
                    'row': row,
                    'col': col,
                    'cell': cell,
                    'color': color,
                    'red_ratio': red_ratio,
                    'black_ratio': black_ratio
                })
                total_pieces += 1
                if color == 'red':
                    red_pieces += 1
                else:
                    black_pieces += 1
    
    print(f"✓ 检测到 {total_pieces} 个棋子")
    print(f"  红色: {red_pieces} 个")
    print(f"  黑色: {black_pieces} 个")
    
    # 第二遍：对每个检测到的棋子进行模板匹配
    print("\n步骤2：模板匹配识别棋子类型")
    print("-"*60)
    
    for detection in piece_detections:
        row = detection['row']
        col = detection['col']
        cell = detection['cell']
        color = detection['color']
        
        # 模板匹配（使用带调试输出的版本）
        piece_name, match_score = match_piece_with_templates_debug(cell, templates, color, row, col)
        
        if piece_name:
            piece_positions[(row, col)] = piece_name
            matched_pieces += 1
            
            # 简化输出
            piece_type = piece_name.split('_')[1]  # 例如 'r_king' -> 'king'
            color_str = '红' if color == 'red' else '黑'
            print(f"  [{row:2d},{col:2d}] {color_str}色 {piece_type:8s} (匹配度: {match_score:.3f})")
        else:
            unmatched_pieces += 1
            print(f"  [{row:2d},{col:2d}] {color}色 未匹配 (最高分: {match_score:.3f})")
        
        # 记录到报告
        detection_report.append({
            'row': row,
            'col': col,
            'color': color,
            'piece_name': piece_name,
            'match_score': match_score,
            'red_ratio': detection['red_ratio'],
            'black_ratio': detection['black_ratio']
        })
    
    print("-"*60)
    print(f"\n识别统计:")
    print(f"  检测到棋子: {total_pieces} 个")
    print(f"  成功匹配: {matched_pieces} 个")
    print(f"  未能匹配: {unmatched_pieces} 个")
    
    return piece_positions, detection_report


def detect_player_side(piece_positions):
    """
    判断玩家阵营（通过红色king的位置）
    
    参数:
        piece_positions: 字典 {(row, col): piece_name}
    
    返回:
        player_side: 'red' 或 'black'
        red_king_row: 红色king所在行号
    """
    # 查找红色king的位置
    red_king_row = None
    for (row, col), piece_name in piece_positions.items():
        if piece_name == 'r_king':
            red_king_row = row
            break
    
    if red_king_row is None:
        print("⚠️  未找到红色king，默认玩家为红方")
        return 'red', None
    
    # 红色king在上方（行号小）→ 玩家是黑方
    # 红色king在下方（行号大）→ 玩家是红方
    if red_king_row <= 5:
        player_side = 'black'
        print(f"✓ 红色king在第{red_king_row}行（上方），玩家是黑方")
    else:
        player_side = 'red'
        print(f"✓ 红色king在第{red_king_row}行（下方），玩家是红方")
    
    return player_side, red_king_row


def generate_fen_from_positions(piece_positions, player_side):
    """
    从棋子位置生成FEN码
    
    参数:
        piece_positions: 字典 {(row, col): piece_name}
        player_side: 'red' 或 'black'
    
    返回:
        fen_string: FEN码
    """
    # FEN映射表
    fen_map = {
        'r_king': 'K',
        'r_guard': 'A',
        'r_bishop': 'B',
        'r_knight': 'N',
        'r_rook': 'R',
        'r_cannon': 'C',
        'r_pawn': 'P',
        'b_king': 'k',
        'b_guard': 'a',
        'b_bishop': 'b',
        'b_knight': 'n',
        'b_rook': 'r',
        'b_cannon': 'c',
        'b_pawn': 'p',
    }
    
    fen_rows = []
    
    for row in range(1, 11):  # 1-10
        fen_row = ""
        empty_count = 0
        
        for col in range(1, 10):  # 1-9
            piece_name = piece_positions.get((row, col))
            
            if piece_name is None:
                # 空格
                empty_count += 1
            else:
                # 有棋子
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                
                fen_char = fen_map.get(piece_name, '?')
                fen_row += fen_char
        
        # 处理行末的空格
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    # 用 '/' 连接各行
    fen_board = '/'.join(fen_rows)
    
    # 添加其他FEN字段
    # 轮到谁走：如果玩家是红方，默认红方先走(w)；如果玩家是黑方，默认黑方先走(b)
    side_to_move = 'w' if player_side == 'red' else 'b'
    fen_string = f"{fen_board} {side_to_move} - - 0 1"
    
    return fen_string


def save_recognition_report(detection_report, piece_positions, player_side, fen_string):
    """
    保存识别报告
    
    参数:
        detection_report: 检测报告列表
        piece_positions: 棋子位置字典
        player_side: 玩家阵营
        fen_string: FEN码
    """
    if not VisionConfig.DEBUG_ENABLED:
        return
    
    report_path = os.path.join(VisionConfig.DEBUG_DIR, "09_recognition_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("棋子识别报告\n")
        f.write("="*100 + "\n\n")
        
        # 识别结果
        f.write("识别结果:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'位置':<10} {'颜色':<8} {'棋子类型':<15} {'匹配度':<10} "
                f"{'红色比例':<12} {'黑色比例':<12}\n")
        f.write("-"*100 + "\n")
        
        for report in detection_report:
            row = report['row']
            col = report['col']
            color_str = '红色' if report['color'] == 'red' else '黑色'
            piece_name = report['piece_name'] if report['piece_name'] else '未匹配'
            match_score = report['match_score']
            red_ratio = report['red_ratio']
            black_ratio = report['black_ratio']
            
            f.write(f"[{row:2d},{col:2d}]    "
                   f"{color_str:<8} "
                   f"{piece_name:<15} "
                   f"{match_score:8.3f}  "
                   f"{red_ratio:10.3f}  "
                   f"{black_ratio:10.3f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # 统计信息
        f.write("="*100 + "\n")
        f.write("统计信息\n")
        f.write("="*100 + "\n\n")
        
        total_detected = len(detection_report)
        matched = sum(1 for r in detection_report if r['piece_name'])
        unmatched = total_detected - matched
        red_count = sum(1 for r in detection_report if r['color'] == 'red')
        black_count = sum(1 for r in detection_report if r['color'] == 'black')
        
        f.write(f"检测到棋子: {total_detected} 个\n")
        f.write(f"  红色: {red_count} 个\n")
        f.write(f"  黑色: {black_count} 个\n")
        f.write(f"成功匹配: {matched} 个\n")
        f.write(f"未能匹配: {unmatched} 个\n\n")
        
        # 玩家阵营
        f.write("="*100 + "\n")
        f.write("玩家阵营\n")
        f.write("="*100 + "\n\n")
        
        player_str = '红方' if player_side == 'red' else '黑方'
        f.write(f"玩家阵营: {player_str}\n\n")
        
        # FEN码
        f.write("="*100 + "\n")
        f.write("FEN码\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{fen_string}\n\n")
        
        # 棋盘布局
        f.write("棋盘布局:\n")
        f.write("-"*100 + "\n")
        parts = fen_string.split()
        if len(parts) > 0:
            rows = parts[0].split('/')
            for i, row in enumerate(rows):
                f.write(f"第{i+1:2d}行: {row}\n")
        
        f.write("-"*100 + "\n\n")
    
    print(f"   💾 已保存: 09_recognition_report.txt")


# ============ FEN码生成 ============
def cells_to_fen(cells, templates):
    """
    将90个格子转换为FEN码
    
    新流程:
    1. 识别所有棋子位置和类型
    2. 判断玩家阵营
    3. 生成FEN码
    
    参数:
        cells: 90个格子图像列表
        templates: 模板字典
    
    返回:
        fen_string (str): FEN码
    """
    # 1. 识别所有棋子
    piece_positions, detection_report = recognize_all_pieces(cells, templates)
    
    # 2. 判断玩家阵营
    print("\n步骤3：判断玩家阵营")
    print("-"*60)
    player_side, red_king_row = detect_player_side(piece_positions)
    
    # 3. 生成FEN码
    print("\n步骤4：生成FEN码")
    print("-"*60)
    fen_string = generate_fen_from_positions(piece_positions, player_side)
    print(f"✓ FEN码: {fen_string}")
    
    # 4. 保存识别报告
    if VisionConfig.DEBUG_ENABLED:
        save_recognition_report(detection_report, piece_positions, player_side, fen_string)
    
    return fen_string


# ============ 主要API ============
def initialize_vision():
    """
    初始化视觉模块（任务1）
    1. 截屏并框选棋盘区域
    2. 加载14个模板
    
    返回:
        success (bool): 是否成功
    """
    print("="*60)
    print("初始化视觉模块")
    print("="*60)
    
    # 1. 框选棋盘区域
    print("\n步骤1：框选棋盘区域")
    print("-"*60)
    selector = RegionSelector()
    bbox = selector.select_region()
    
    if bbox is None:
        print("❌ 未选择区域")
        return False
    
    vision_state.bbox = bbox
    print(f"✓ 棋盘区域已保存: {bbox}")
    
    # 2. 加载模板
    print("\n步骤2：加载模板")
    print("-"*60)
    templates = load_templates()
    
    if len(templates) == 0:
        print("❌ 未加载任何模板")
        return False
    
    vision_state.templates = templates
    print(f"✓ 已加载 {len(templates)} 个模板")
    
    vision_state.initialized = True
    
    print("\n" + "="*60)
    print("✅ 视觉模块初始化完成")
    print("="*60)
    
    return True


def debug_detect_pieces_and_colors(cells, board_image):
    """
    调试函数：检测每个格子是否有棋子及颜色
    
    参数:
        cells: 90个格子图像列表
        board_image: 裁剪后的棋盘图像
    
    返回:
        detection_results: 检测结果列表
    """
    print("\n" + "="*60)
    print("调试：检测棋子和颜色")
    print("="*60)
    
    detection_results = []
    piece_count = 0
    red_count = 0
    black_count = 0
    high_contrast_no_piece = 0  # 对比度高但没有棋子（楚河汉界）
    
    for idx, cell in enumerate(cells):
        row = idx // 9 + 1
        col = idx % 9 + 1
        
        # 1. 检测是否为空（对比度）
        is_empty = is_empty_cell(cell)
        
        if not is_empty:
            # 2. 检测颜色
            color, red_ratio, black_ratio = detect_piece_color(cell)
            
            # 3. 根据颜色判断是否真的有棋子
            if color == 'red':
                has_piece = True
                piece_count += 1
                red_count += 1
                marker = 'R'
            elif color == 'black':
                has_piece = True
                piece_count += 1
                black_count += 1
                marker = 'B'
            else:
                # 对比度高但颜色不符合（楚河汉界）
                has_piece = False
                marker = ''
                high_contrast_no_piece += 1
        else:
            has_piece = False
            color = 'none'
            marker = ''
            red_ratio = 0
            black_ratio = 0
        
        detection_results.append({
            'row': row,
            'col': col,
            'has_piece': has_piece,
            'color': color,
            'marker': marker,
            'red_ratio': red_ratio if not is_empty else 0,
            'black_ratio': black_ratio if not is_empty else 0
        })
    
    print(f"\n检测统计:")
    print(f"   总格子数: {len(cells)}")
    print(f"   有棋子: {piece_count}")
    print(f"     红色棋子: {red_count}")
    print(f"     黑色棋子: {black_count}")
    print(f"   没有棋子: {len(cells) - piece_count}")
    print(f"     对比度低: {len(cells) - piece_count - high_contrast_no_piece}")
    print(f"     对比度高但无棋子（楚河汉界等）: {high_contrast_no_piece}")
    
    # 保存标注后的棋盘图像
    if VisionConfig.DEBUG_ENABLED:
        annotated_board = debug_annotate_board(board_image, detection_results)
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, "06_piece_detection.png"), annotated_board)
        print(f"\n   💾 已保存: 06_piece_detection.png")
        
        # 保存检测报告
        debug_save_detection_report(detection_results)
    
    return detection_results


def debug_annotate_board(board_image, detection_results):
    """
    调试函数：在棋盘图像上标注检测结果
    
    参数:
        board_image: 棋盘图像
        detection_results: 检测结果列表
    
    返回:
        annotated_board: 标注后的棋盘图像
    """
    annotated_board = board_image.copy()
    
    # 计算格子尺寸
    rows, cols = 10, 9
    cell_height = board_image.shape[0] // rows
    cell_width = board_image.shape[1] // cols
    
    # 标注每个格子
    for result in detection_results:
        if result['marker']:
            row = result['row']
            col = result['col']
            marker = result['marker']
            
            # 计算格子中心位置
            center_x = int((col - 0.5) * cell_width)
            center_y = int((row - 0.5) * cell_height)
            
            # 绘制标记
            if marker == 'R':
                color = (0, 0, 255)  # 红色
                text = 'R'
            elif marker == 'B':
                color = (0, 0, 0)    # 黑色
                text = 'B'
            else:
                color = (128, 128, 128)  # 灰色
                text = '?'
            
            # 绘制圆形背景
            cv2.circle(annotated_board, (center_x, center_y), 15, (255, 255, 255), -1)
            cv2.circle(annotated_board, (center_x, center_y), 15, color, 2)
            
            # 绘制文字
            cv2.putText(annotated_board, text, (center_x - 8, center_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated_board


def debug_save_detection_report(detection_results):
    """
    调试函数：保存检测报告
    
    参数:
        detection_results: 检测结果列表
    """
    report_path = os.path.join(VisionConfig.DEBUG_DIR, "07_detection_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("棋子检测报告\n")
        f.write("="*80 + "\n\n")
        
        # 有棋子的格子
        f.write("有棋子的格子:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'位置':<10} {'颜色':<10} {'标记':<6}\n")
        f.write("-"*80 + "\n")
        
        piece_results = [r for r in detection_results if r['has_piece']]
        for result in piece_results:
            color_str = '红色' if result['color'] == 'red' else '黑色'
            f.write(f"[{result['row']:2d},{result['col']:2d}]    "
                   f"{color_str:<10} "
                   f"{result['marker']:<6}\n")
        
        f.write("-"*80 + "\n")
        f.write(f"小计: {len(piece_results)} 个格子有棋子\n\n")
        
        # 统计信息
        f.write("="*80 + "\n")
        f.write("统计信息\n")
        f.write("="*80 + "\n\n")
        
        piece_count = sum(1 for r in detection_results if r['has_piece'])
        red_count = sum(1 for r in detection_results if r['marker'] == 'R')
        black_count = sum(1 for r in detection_results if r['marker'] == 'B')
        empty_count = len(detection_results) - piece_count
        
        f.write(f"总格子数: {len(detection_results)}\n")
        f.write(f"  有棋子: {piece_count}\n")
        f.write(f"    红色棋子: {red_count}\n")
        f.write(f"    黑色棋子: {black_count}\n")
        f.write(f"  空格子: {empty_count}\n\n")
        
        # 按行统计
        f.write("="*80 + "\n")
        f.write("按行统计\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'行号':<6} {'有棋子数':<10} {'空格子数':<10}\n")
        f.write("-"*80 + "\n")
        
        for row in range(1, 11):
            row_results = [r for r in detection_results if r['row'] == row]
            row_piece_count = sum(1 for r in row_results if r['has_piece'])
            row_empty_count = len(row_results) - row_piece_count
            
            f.write(f"第{row:2d}行  "
                   f"{row_piece_count:<10} "
                   f"{row_empty_count:<10}\n")
        
        f.write("-"*80 + "\n\n")
    
    print(f"   💾 已保存: 07_detection_report.txt")


def capture_and_recognize():
    """
    截取棋盘并识别为FEN码（任务2）
    
    返回:
        fen_string (str): FEN码，失败返回None
    """
    if not vision_state.initialized:
        print("❌ 视觉模块未初始化")
        return None
    
    print("="*60)
    print("截取棋盘并识别")
    print("="*60)
    
    # 创建调试目录
    if VisionConfig.DEBUG_ENABLED:
        os.makedirs(VisionConfig.DEBUG_DIR, exist_ok=True)
    
    # 1. 截取棋盘区域
    print("\n步骤1：截取棋盘区域")
    print("-"*60)
    x1, y1, x2, y2 = vision_state.bbox
    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    board_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print(f"✓ 已截取棋盘: {board_image.shape[1]}x{board_image.shape[0]}")
    
    # 调试：保存原始截图
    if VisionConfig.DEBUG_ENABLED:
        cv2.imwrite(os.path.join(VisionConfig.DEBUG_DIR, "01_original_screenshot.png"), board_image)
        print(f"   💾 已保存: 01_original_screenshot.png")
    
    # 2. 分割为90个格子
    print("\n步骤2：分割为90个格子")
    print("-"*60)
    cells = split_chessboard_to_cells(board_image, debug_prefix="")
    
    if len(cells) != 90:
        print(f"❌ 格子数量错误: {len(cells)} (应为90)")
        return None
    
    print(f"✓ 已分割为 {len(cells)} 个格子")
    
    # 调试：保存部分格子样本
    if VisionConfig.DEBUG_ENABLED:
        sample_cells_dir = os.path.join(VisionConfig.DEBUG_DIR, "sample_cells")
        os.makedirs(sample_cells_dir, exist_ok=True)
        
        # 保存四个角的格子和中心格子
        sample_indices = [
            (0, "01-01_top_left"),      # 左上角
            (8, "01-09_top_right"),     # 右上角
            (81, "10-01_bottom_left"),  # 左下角
            (89, "10-09_bottom_right"), # 右下角
            (40, "05-05_center"),       # 中心
            (3, "01-04"),
            (29, "04-03"),
            (67, "08-05"),
            (42, "05-07"),
            (4, "01-05"),
        ]
        
        for idx, name in sample_indices:
            if idx < len(cells):
                cv2.imwrite(os.path.join(sample_cells_dir, f"{name}.png"), cells[idx])
        
        print(f"   💾 已保存样本格子到: sample_cells/")
    
    # 调试：检测棋子和颜色
    if VisionConfig.DEBUG_ENABLED:
        # 需要获取裁剪后的棋盘图像
        mask = extract_board_color_mask(board_image)
        board_rect = find_chessboard_contour(mask)
        
        if board_rect:
            x, y, w, h = board_rect
            
            expanded_x = x - VisionConfig.EXPAND_LEFT
            expanded_y = y - VisionConfig.EXPAND_TOP
            expanded_w = w + VisionConfig.EXPAND_LEFT + VisionConfig.EXPAND_RIGHT
            expanded_h = h + VisionConfig.EXPAND_TOP + VisionConfig.EXPAND_BOTTOM
            
            expanded_x = max(0, expanded_x)
            expanded_y = max(0, expanded_y)
            
            if expanded_y + expanded_h > board_image.shape[0]:
                expanded_h = board_image.shape[0] - expanded_y
            
            if expanded_x + expanded_w > board_image.shape[1]:
                expanded_w = board_image.shape[1] - expanded_x
            
            cropped_board = board_image[expanded_y:expanded_y+expanded_h, 
                                       expanded_x:expanded_x+expanded_w]
            
            debug_detect_pieces_and_colors(cells, cropped_board)
    
    # 3. 识别并生成FEN码
    print("\n" + "="*60)
    print("识别棋子并生成FEN码")
    print("="*60)
    fen_string = cells_to_fen(cells, vision_state.templates)
    
    print("\n" + "="*60)
    print("✅ 识别完成")
    print("="*60)
    print(f"\n📁 调试文件保存在: {VisionConfig.DEBUG_DIR}/")
    
    return fen_string

