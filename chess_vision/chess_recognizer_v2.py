"""
象棋棋盘识别 - 改进版
专门针对天天象棋等实际场景优化
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from PIL import Image


class ChessboardRecognizer:
    """象棋棋盘识别器 - 改进版"""
    
    def __init__(self, debug=False):
        """
        初始化识别器
        
        Args:
            debug: 是否开启调试模式（保存中间图像）
        """
        self.debug = debug
        self.debug_images = {}
        
    def save_debug_image(self, name: str, image: np.ndarray):
        """保存调试图像"""
        if self.debug:
            self.debug_images[name] = image.copy()
            cv2.imwrite(f"debug_{name}.png", image)
            print(f"💾 已保存调试图像: debug_{name}.png")
    
    def detect_chessboard_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        高级棋盘检测 - 使用多种方法
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (棋盘图像, (x, y, width, height))
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.save_debug_image("01_gray", gray)
        
        # 方法1：边缘检测 + 轮廓
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        self.save_debug_image("02_edges", edges)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        self.save_debug_image("03_dilated", dilated)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("未检测到棋盘轮廓")
        
        # 筛选合适的轮廓（面积、长宽比）
        valid_contours = []
        img_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.1:  # 至少占10%的面积
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # 象棋棋盘宽高比约为 0.9-1.1
            if 0.7 < aspect_ratio < 1.3:
                valid_contours.append((contour, area, (x, y, w, h)))
        
        if not valid_contours:
            raise Exception("未找到符合条件的棋盘")
        
        # 选择面积最大的
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        _, _, (x, y, w, h) = valid_contours[0]
        
        # 绘制检测结果
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        self.save_debug_image("04_detected_board", debug_img)
        
        # 裁剪棋盘
        board_image = image[y:y+h, x:x+w]
        
        return board_image, (x, y, w, h)
    
    def segment_board_adaptive(self, board_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        自适应分割棋盘 - 检测网格线
        
        Args:
            board_image: 棋盘图像
            
        Returns:
            [(格子图像, (row, col)), ...]
        """
        h, w = board_image.shape[:2]
        
        # 简单均匀分割（后续可以改进为检测网格线）
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
                cells.append((cell, (i, j)))
        
        return cells
    
    def detect_piece_advanced(self, cell: np.ndarray) -> Optional[Dict]:
        """
        高级棋子检测
        
        Args:
            cell: 格子图像
            
        Returns:
            {'has_piece': bool, 'color': str} 或 None
        """
        # 检测是否有棋子（对比度）
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, stddev = cv2.meanStdDev(gray)
        contrast = stddev[0][0]
        
        if contrast < 25:  # 降低阈值，更敏感
            return None
        
        # 检测颜色
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # 红色检测（更宽松的范围）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([155, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 黑色检测
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        total_pixels = cell.shape[0] * cell.shape[1]
        red_ratio = cv2.countNonZero(mask_red) / total_pixels
        black_ratio = cv2.countNonZero(mask_black) / total_pixels
        
        # 降低阈值
        if red_ratio > 0.03:
            return {'has_piece': True, 'color': 'red'}
        elif black_ratio > 0.03:
            return {'has_piece': True, 'color': 'black'}
        
        return None
    
    def generate_fen(self, pieces: List[Dict]) -> str:
        """
        生成FEN码
        
        Args:
            pieces: 棋子列表
            
        Returns:
            FEN字符串
        """
        # 创建10x9的棋盘
        board = [['.' for _ in range(9)] for _ in range(10)]
        
        # 放置棋子（暂时都用兵/卒）
        for piece in pieces:
            row, col = piece['position']
            if piece['color'] == 'red':
                board[row][col] = 'P'  # 红兵
            else:
                board[row][col] = 'p'  # 黑卒
        
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
    
    def visualize_detection(self, board_image: np.ndarray, pieces: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            board_image: 棋盘图像
            pieces: 检测到的棋子
            
        Returns:
            标注后的图像
        """
        result = board_image.copy()
        h, w = board_image.shape[:2]
        cell_height = h // 10
        cell_width = w // 9
        
        for piece in pieces:
            row, col = piece['position']
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2
            
            color = (0, 0, 255) if piece['color'] == 'red' else (255, 0, 0)
            cv2.circle(result, (x, y), 10, color, -1)
            cv2.putText(result, piece['color'][0].upper(), (x-5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def recognize(self, image: np.ndarray) -> Dict:
        """
        识别棋盘并返回FEN码
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            {'success': bool, 'fen': str, 'elapsed_ms': int, ...}
        """
        import time
        start_time = time.time()
        
        try:
            print("=" * 60)
            print("开始识别...")
            print("=" * 60)
            
            # 1. 检测棋盘
            print("🔍 步骤1: 检测棋盘...")
            board_image, board_rect = self.detect_chessboard_advanced(image)
            print(f"   ✓ 棋盘位置: {board_rect}")
            
            # 2. 分割格子
            print("✂️  步骤2: 分割格子...")
            cells = self.segment_board_adaptive(board_image)
            print(f"   ✓ 分割完成: {len(cells)} 个格子")
            
            # 3. 识别棋子
            print("🎯 步骤3: 识别棋子...")
            pieces = []
            for cell, (row, col) in cells:
                result = self.detect_piece_advanced(cell)
                if result:
                    pieces.append({
                        'position': (row, col),
                        'color': result['color'],
                        'type': 'pawn'
                    })
            
            print(f"   ✓ 检测到 {len(pieces)} 个棋子")
            
            # 统计红黑棋子数量
            red_count = sum(1 for p in pieces if p['color'] == 'red')
            black_count = sum(1 for p in pieces if p['color'] == 'black')
            print(f"   - 红方: {red_count} 个")
            print(f"   - 黑方: {black_count} 个")
            
            # 4. 生成FEN码
            print("📋 步骤4: 生成FEN码...")
            fen = self.generate_fen(pieces)
            
            # 5. 可视化
            if self.debug:
                vis_image = self.visualize_detection(board_image, pieces)
                self.save_debug_image("05_final_result", vis_image)
            
            elapsed = int((time.time() - start_time) * 1000)
            
            print("=" * 60)
            print(f"✅ 识别完成！耗时: {elapsed}ms")
            print("=" * 60)
            
            return {
                'success': True,
                'fen': fen,
                'elapsed_ms': elapsed,
                'detected_pieces': len(pieces),
                'red_pieces': red_count,
                'black_pieces': black_count,
                'board_rect': board_rect
            }
            
        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            print(f"❌ 识别失败: {e}")
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
    print("象棋识别测试 - 改进版")
    print("=" * 60)
    print()
    print("请选择测试方式:")
    print("  1) 截取全屏")
    print("  2) 截取指定区域")
    print("  3) 从文件读取")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    # 创建识别器（开启调试模式）
    recognizer = ChessboardRecognizer(debug=True)
    
    if choice == '1':
        print("\n将在3秒后截取全屏...")
        time.sleep(3)
        screenshot = ImageGrab.grab()
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
    elif choice == '2':
        print("\n请输入截图区域 (left, top, right, bottom):")
        left = int(input("  left: "))
        top = int(input("  top: "))
        right = int(input("  right: "))
        bottom = int(input("  bottom: "))
        print("\n将在3秒后截取指定区域...")
        time.sleep(3)
        screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
    elif choice == '3':
        image_path = input("\n请输入图片路径: ").strip()
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            exit(1)
    else:
        print("无效选择")
        exit(1)
    
    # 保存原始图像
    cv2.imwrite("debug_00_original.png", image)
    print("💾 已保存原始图像: debug_00_original.png")
    
    # 识别
    result = recognizer.recognize(image)
    
    if result['success']:
        print(f"\n📋 FEN码: {result['fen']}")
        print(f"\n💡 提示: 调试图像已保存到当前目录，可以查看识别过程")
    else:
        print(f"\n查看调试图像以了解失败原因")

