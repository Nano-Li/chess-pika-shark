"""
鲨鱼象棋控制模块：UI自动化
"""

import win32gui
import win32con
import win32api
import win32clipboard
import time


# ============ 配置参数 ============
class SharkConfig:
    # 窗口类名和标题
    WINDOW_CLASS = "TForm1"
    WINDOW_TITLE_KEYWORD = "鲨鱼象棋"
    
    # 点击位置（相对于窗口左上角）
    CLICK_MENU_BUTTON = (80, 50)      # "棋局"按钮
    CLICK_PASTE_BUTTON = (120, 196)   # "粘贴局面/棋谱"按钮
    
    # 快捷键（引擎控制）
    HOTKEY_ENGINE_RED = (0x11, 0x12, 0x52)   # Ctrl+Alt+R
    HOTKEY_ENGINE_BLACK = (0x11, 0x12, 0x42)  # Ctrl+Alt+B
    
    # 延迟时间（秒）
    DELAY_ACTIVATE = 0.3
    DELAY_MENU = 0.2
    DELAY_CLICK = 0.05
    DELAY_HOTKEY = 0.05


# ============ 全局状态 ============
class SharkState:
    """鲨鱼象棋状态"""
    def __init__(self):
        self.main_hwnd = None
        self.red_engine_enabled = False   # 红色引擎状态
        self.black_engine_enabled = False  # 黑色引擎状态
        self.initialized = False


shark_state = SharkState()


# ============ 窗口操作 ============
def find_shark_window():
    """
    查找鲨鱼象棋主窗口
    返回: hwnd 或 None
    """
    try:
        hwnd = win32gui.FindWindow(SharkConfig.WINDOW_CLASS, None)
        
        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            if SharkConfig.WINDOW_TITLE_KEYWORD in title:
                print(f"✓ 找到鲨鱼象棋窗口")
                print(f"  句柄: {hwnd}")
                print(f"  标题: {title}")
                return hwnd
        
        print("❌ 未找到鲨鱼象棋窗口")
        return None
        
    except Exception as e:
        print(f"❌ 查找窗口失败: {e}")
        return None


def activate_window(hwnd):
    """
    激活窗口并置顶
    """
    try:
        # 如果窗口最小化，先恢复
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        # 显示窗口
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        
        # 置顶窗口
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                             win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        
        # 取消置顶（但保持在最前）
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                             win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        
        # 设置为前台窗口
        win32gui.SetForegroundWindow(hwnd)
        
        print("✓ 窗口已激活")
        return True
        
    except Exception as e:
        print(f"❌ 激活窗口失败: {e}")
        return False


def is_window_visible(hwnd):
    """
    检查窗口是否可见
    """
    try:
        return win32gui.IsWindowVisible(hwnd)
    except:
        return False


# ============ 剪贴板操作 ============
def set_clipboard(text):
    """
    设置剪贴板内容
    """
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        print(f"✓ 已复制到剪贴板: {text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ 复制失败: {e}")
        return False


# ============ 鼠标操作 ============
def click_by_relative_position(hwnd, rel_x, rel_y):
    """
    通过相对窗口的位置点击
    """
    try:
        # 获取窗口位置
        rect = win32gui.GetWindowRect(hwnd)
        
        # 计算绝对坐标
        abs_x = rect[0] + rel_x
        abs_y = rect[1] + rel_y
        
        # 移动鼠标并点击
        win32api.SetCursorPos((abs_x, abs_y))
        time.sleep(SharkConfig.DELAY_CLICK)
        
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(SharkConfig.DELAY_CLICK)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
        return True
        
    except Exception as e:
        print(f"❌ 点击失败: {e}")
        return False


# ============ 键盘操作 ============
def send_hotkey(hwnd, vk_codes):
    """
    发送快捷键
    参数: vk_codes - 虚拟键码列表，例如 (VK_CONTROL, VK_ALT, VK_R)
    """
    try:
        # 激活窗口
        activate_window(hwnd)
        time.sleep(SharkConfig.DELAY_ACTIVATE)
        
        # 按下所有键
        for vk in vk_codes:
            win32api.keybd_event(vk, 0, 0, 0)
            time.sleep(SharkConfig.DELAY_HOTKEY)
        
        # 释放所有键（顺序相反）
        for vk in reversed(vk_codes):
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(SharkConfig.DELAY_HOTKEY)
        
        return True
        
    except Exception as e:
        print(f"❌ 发送快捷键失败: {e}")
        return False


# ============ FEN码相关 ============
def detect_side_from_fen(fen_string):
    """
    从FEN码判断当前轮到哪方走棋
    
    FEN格式: "棋盘布局 w/b - - 0 1"
    w = 红方走棋 (white)
    b = 黑方走棋 (black)
    
    返回: 'red' 或 'black'
    """
    try:
        parts = fen_string.split()
        if len(parts) >= 2:
            side = parts[1].lower()
            if side == 'w':
                return 'red'
            elif side == 'b':
                return 'black'
        
        # 默认红方先走
        print("⚠️  无法从FEN码判断走棋方，默认为红方")
        return 'red'
        
    except Exception as e:
        print(f"⚠️  解析FEN码失败: {e}，默认为红方")
        return 'red'


def paste_fen_to_shark(hwnd, fen_string):
    """
    粘贴FEN码到鲨鱼象棋
    
    流程:
    1. 复制FEN到剪贴板
    2. 点击"棋局"按钮
    3. 点击"粘贴局面/棋谱"按钮
    """
    print("\n" + "="*60)
    print("粘贴FEN码到鲨鱼象棋")
    print("="*60)
    
    # 1. 复制FEN到剪贴板
    print("\n步骤1：复制FEN到剪贴板")
    if not set_clipboard(fen_string):
        return False
    
    # 2. 激活窗口
    print("\n步骤2：激活窗口")
    if not activate_window(hwnd):
        return False
    
    time.sleep(SharkConfig.DELAY_ACTIVATE)
    
    # 3. 点击"棋局"按钮
    print("\n步骤3：点击'棋局'按钮")
    rel_x, rel_y = SharkConfig.CLICK_MENU_BUTTON
    if not click_by_relative_position(hwnd, rel_x, rel_y):
        return False
    
    # 4. 等待菜单弹出
    time.sleep(SharkConfig.DELAY_MENU)
    
    # 5. 点击"粘贴局面/棋谱"按钮
    print("\n步骤4：点击'粘贴局面/棋谱'按钮")
    rel_x, rel_y = SharkConfig.CLICK_PASTE_BUTTON
    if not click_by_relative_position(hwnd, rel_x, rel_y):
        return False
    
    print("\n✅ FEN码已粘贴")
    return True


# ============ 引擎控制 ============
def start_engine(hwnd, side):
    """
    开启引擎分析
    
    参数:
        hwnd: 窗口句柄
        side: 'red' 或 'black'
    """
    print("\n" + "="*60)
    print(f"开启{'红色' if side == 'red' else '黑色'}引擎")
    print("="*60)
    
    # 选择快捷键
    if side == 'red':
        hotkey = SharkConfig.HOTKEY_ENGINE_RED
        print(f"\n快捷键: Ctrl+Alt+R")
    else:
        hotkey = SharkConfig.HOTKEY_ENGINE_BLACK
        print(f"\n快捷键: Ctrl+Alt+B")
    
    # 发送快捷键
    if not send_hotkey(hwnd, hotkey):
        return False
    
    # 更新状态
    if side == 'red':
        shark_state.red_engine_enabled = True
    else:
        shark_state.black_engine_enabled = True
    
    print(f"\n✅ {'红色' if side == 'red' else '黑色'}引擎已开启")
    return True


def stop_engine(hwnd, side):
    """
    关闭引擎分析
    
    参数:
        hwnd: 窗口句柄
        side: 'red' 或 'black'
    """
    print("\n" + "="*60)
    print(f"关闭{'红色' if side == 'red' else '黑色'}引擎")
    print("="*60)
    
    # 选择快捷键（Toggle按钮，再按一次就关闭）
    if side == 'red':
        hotkey = SharkConfig.HOTKEY_ENGINE_RED
        print(f"\n快捷键: Ctrl+Alt+R")
    else:
        hotkey = SharkConfig.HOTKEY_ENGINE_BLACK
        print(f"\n快捷键: Ctrl+Alt+B")
    
    # 发送快捷键
    if not send_hotkey(hwnd, hotkey):
        return False
    
    # 更新状态
    if side == 'red':
        shark_state.red_engine_enabled = False
    else:
        shark_state.black_engine_enabled = False
    
    print(f"\n✅ {'红色' if side == 'red' else '黑色'}引擎已关闭")
    return True


# ============ 主要API ============
def initialize_shark_control():
    """
    初始化鲨鱼象棋控制模块
    
    功能:
    1. 查找鲨鱼象棋窗口
    2. 检查窗口是否可见
    3. 初始化引擎状态（默认都未开启）
    
    返回: success (bool)
    """
    print("="*60)
    print("初始化鲨鱼象棋控制模块")
    print("="*60)
    
    # 1. 查找窗口
    print("\n步骤1：查找鲨鱼象棋窗口")
    print("-"*60)
    hwnd = find_shark_window()
    
    if hwnd is None:
        print("❌ 未找到鲨鱼象棋窗口，请先打开鲨鱼象棋")
        return False
    
    shark_state.main_hwnd = hwnd
    
    # 2. 检查窗口是否可见
    print("\n步骤2：检查窗口状态")
    print("-"*60)
    if not is_window_visible(hwnd):
        print("⚠️  窗口不可见，尝试激活...")
        if not activate_window(hwnd):
            print("❌ 无法激活窗口")
            return False
    else:
        print("✓ 窗口可见")
    
    # 3. 初始化引擎状态
    print("\n步骤3：初始化引擎状态")
    print("-"*60)
    shark_state.red_engine_enabled = False
    shark_state.black_engine_enabled = False
    print("✓ 引擎状态已初始化（红色=关闭，黑色=关闭）")
    
    shark_state.initialized = True
    
    print("\n" + "="*60)
    print("✅ 鲨鱼象棋控制模块初始化完成")
    print("="*60)
    
    return True


def send_fen_and_start_engine(fen_string):
    """
    发送FEN码并开启引擎分析
    
    流程:
    1. 粘贴FEN码到鲨鱼象棋
    2. 从FEN码判断当前轮到哪方走棋
    3. 开启对应方的引擎
    
    返回: success (bool)
    """
    if not shark_state.initialized:
        print("❌ 鲨鱼象棋控制模块未初始化")
        return False
    
    hwnd = shark_state.main_hwnd
    
    # 1. 粘贴FEN码
    if not paste_fen_to_shark(hwnd, fen_string):
        return False
    
    time.sleep(0.3)
    
    # 2. 判断当前走棋方
    side = detect_side_from_fen(fen_string)
    print(f"\n✓ 当前轮到: {'红方' if side == 'red' else '黑方'}")
    
    # 3. 开启对应引擎
    if not start_engine(hwnd, side):
        return False
    
    return True


def stop_all_engines():
    """
    关闭所有已开启的引擎
    """
    if not shark_state.initialized:
        print("❌ 鲨鱼象棋控制模块未初始化")
        return False
    
    hwnd = shark_state.main_hwnd
    success = True
    
    # 关闭红色引擎
    if shark_state.red_engine_enabled:
        if not stop_engine(hwnd, 'red'):
            success = False
    
    # 关闭黑色引擎
    if shark_state.black_engine_enabled:
        if not stop_engine(hwnd, 'black'):
            success = False
    
    if not shark_state.red_engine_enabled and not shark_state.black_engine_enabled:
        print("\n✓ 没有已开启的引擎")
    
    return success

