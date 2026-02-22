"""
象棋棋盘识别 - Python版本
基于OpenCV Python，移植自xiangqi-analysis项目
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import os


class ChessboardRecognizer:
    """象棋棋盘识别器"""
    
    def __init__(self, templates_dir=None):
        """
        初始化识别器
        
        Args:
            templates_dir: 棋子模板图片目录
        """
        self.templates = {}
        self.templates_dir = templates_dir
        
        if templates_dir and os.path.exists(templates_dir):
            self._load_templates()
        else:
            print("⚠️  警告：未找到模板文件，将使用简化识别")
    
    def _load_templates(self):
        """加载棋子模板"""
        print("⏳ 正在加载棋子模板...")
        
        # 棋子类型：红方和黑方各7种
        piece_types = ['king', 'guard', 'bishop', 'knight', 'rook', 'cannon', 'pawn']
        colors = ['red', 'black']
        
        for color in colors:
            for piece_type in piece_types:
                template_path = os.path.join(
                    self.templates_dir, 
                    f"{color}_{piece_type}.png"
                )
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    self.templates[f"{color}_{piece_type}"] = template
        
        print(f"✅ 已加载 {len(self.templates)} 个模板")
    
    def detect_chessboard(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        检测棋盘位置
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (棋盘图像, (x, y, width, height))
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀和腐蚀
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("未检测到棋盘")
        
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 裁剪棋盘区域
        board_image = image[y:y+h, x:x+w]
        
        return board_image, (x, y, w, h)
    
    def segment_board(self, board_image: np.ndarray) -> List[np.ndarray]:
        """
        将棋盘分割成90个格子 (10行 x 9列)
        
        Args:
            board_image: 棋盘图像
            
        Returns:
            90个格子图像的列表
        """
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
                cells.append(cell)
        
        return cells
    
    def detect_piece_in_cell(self, cell: np.ndarray) -> bool:
        """检测格子中是否有棋子"""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, stddev = cv2.meanStdDev(gray)
        contrast = stddev[0][0]
        return contrast > 30
    
    def detect_piece_color(self, cell: np.ndarray) -> str:
        """检测棋子颜色"""
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # 红色范围
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 120])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 黑色范围
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        total_pixels = cell.shape[0] * cell.shape[1]
        red_ratio = cv2.countNonZero(mask_red) / total_pixels
        black_ratio = cv2.countNonZero(mask_black) / total_pixels
        
        if red_ratio > 0.05:
            return 'red'
        elif black_ratio > 0.05:
            return 'black'
        return 'unknown'
    
    def generate_fen(self, pieces: List[Dict]) -> str:
        """
        生成FEN码
        
        Args:
            pieces: 棋子列表，每个元素包含 {position: (row, col), color: str, type: str}
            
        Returns:
            FEN字符串
        """
        # 创建10x9的棋盘
        board = [['.' for _ in range(9)] for _ in range(10)]
        
        # 放置棋子
        piece_map = {
            'king': 'k', 'guard': 'a', 'bishop': 'b',
            'knight': 'n', 'rook': 'r', 'cannon': 'c', 'pawn': 'p'
        }
        
        for piece in pieces:
            row, col = piece['position']
            piece_type = piece_map.get(piece['type'], 'p')
            if piece['color'] == 'red':
                piece_type = piece_type.upper()
            board[row][col] = piece_type
        
        # 生成FEN字符串
        fen_rows = []
        for row in board:
            fen_row = ''
            empty_count = 0
            for cell in row:
                if cell == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        
        fen = '/'.join(fen_rows) + ' w - - 0 1'
        return fen
    
    def recognize(self, image: np.ndarray) -> Dict:
        """
        识别棋盘并返回FEN码
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            {'success': bool, 'fen': str, 'elapsed_ms': int}
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 检测棋盘
            print("🔍 检测棋盘...")
            board_image, board_rect = self.detect_chessboard(image)
            
            # 2. 分割格子
            print("✂️  分割格子...")
            cells = self.segment_board(board_image)
            
            # 3. 识别棋子
            print("🎯 识别棋子...")
            pieces = []
            for idx, cell in enumerate(cells):
                row = idx // 9
                col = idx % 9
                
                if self.detect_piece_in_cell(cell):
                    color = self.detect_piece_color(cell)
                    if color != 'unknown':
                        pieces.append({
                            'position': (row, col),
                            'color': color,
                            'type': 'pawn'  # 简化版本，暂时都识别为兵
                        })
            
            # 4. 生成FEN码
            print("📋 生成FEN码...")
            fen = self.generate_fen(pieces)
            
            elapsed = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'fen': fen,
                'elapsed_ms': elapsed,
                'detected_pieces': len(pieces)
            }
            
        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            return {
                'success': False,
                'error': str(e),
                'elapsed_ms': elapsed
            }


# 测试代码
if __name__ == "__main__":
    from PIL import ImageGrab
    import time
    
    print("=" * 60)
    print("象棋识别测试 - Python版本")
    print("=" * 60)
    
    # 创建识别器
    recognizer = ChessboardRecognizer()
    
    # 截图
    print("\n将在3秒后截取全屏...")
    time.sleep(3)
    screenshot = ImageGrab.grab()
    
    # 转换为OpenCV格式
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # 识别
    result = recognizer.recognize(image)
    
    if result['success']:
        print(f"\n✅ 识别成功")
        print(f"⏱️  耗时: {result['elapsed_ms']}ms")
        print(f"🎯 检测到 {result['detected_pieces']} 个棋子")
        print(f"📋 FEN码: {result['fen']}")
    else:
        print(f"\n❌ 识别失败: {result['error']}")

