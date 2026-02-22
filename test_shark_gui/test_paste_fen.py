"""
测试脚本：自动粘贴FEN棋谱到鲨鱼象棋
功能：复制FEN码，然后点击"粘贴局面/棋谱"按钮
"""

import win32gui
import win32con
import win32api
import win32clipboard
import time
import sys
import ctypes


def is_admin():
    """检查是否有管理员权限"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """请求管理员权限并重启程序"""
    if not is_admin():
        print("检测到需要管理员权限，正在请求提升权限...")
        try:
            # 重新运行程序，请求管理员权限
            ctypes.windll.shell32.ShellExecuteW(
                None, 
                "runas",  # 请求提升权限
                sys.executable,  # python.exe路径
                " ".join(sys.argv),  # 脚本参数
                None, 
                1  # SW_SHOWNORMAL
            )
            sys.exit()
        except Exception as e:
            print(f"✗ 请求管理员权限失败: {e}")
            print("请右键点击脚本，选择'以管理员身份运行'")
            sys.exit(1)
    else:
        print("✓ 已获得管理员权限")


def find_main_window():
    """查找鲨鱼象棋主窗口"""
    # 根据类名和标题查找
    hwnd = win32gui.FindWindow("TForm1", None)
    
    if hwnd:
        title = win32gui.GetWindowText(hwnd)
        if "鲨鱼象棋" in title:
            print(f"✓ 找到主窗口")
            print(f"  句柄: {hwnd}")
            print(f"  标题: {title}")
            return hwnd
    
    print("✗ 未找到鲨鱼象棋窗口")
    return None


def set_clipboard(text):
    """设置剪贴板内容"""
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        print(f"✓ 已复制到剪贴板: {text}")
        return True
    except Exception as e:
        print(f"✗ 复制失败: {e}")
        return False


def click_by_relative_position(main_hwnd, rel_x, rel_y):
    """通过相对窗口的位置点击"""
    try:
        # 获取窗口位置
        rect = win32gui.GetWindowRect(main_hwnd)
        
        # 计算绝对坐标
        abs_x = rect[0] + rel_x
        abs_y = rect[1] + rel_y
        
        print(f"✓ 点击位置 - 相对: ({rel_x}, {rel_y}) | 绝对: ({abs_x}, {abs_y})")
        
        # 移动鼠标并点击
        win32api.SetCursorPos((abs_x, abs_y))
        time.sleep(0.05)
        
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
        print("✓ 已点击")
        return True
        
    except Exception as e:
        print(f"✗ 点击失败: {e}")
        return False


def click_paste_button_by_hwnd(parent_hwnd):
    """【旧方法】通过句柄点击粘贴按钮"""
    try:
        # 查找类名为 #32768 的子控件
        hwnd = win32gui.FindWindowEx(parent_hwnd, 0, '#32768', None)
        
        if hwnd:
            print(f"✓ 找到粘贴按钮")
            print(f"  句柄: {hwnd}")
            
            # 点击按钮
            win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, 0, 0)
            time.sleep(0.05)
            win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, 0)
            
            print("✓ 已点击按钮")
            return True
        else:
            print("✗ 未找到粘贴按钮（类名 #32768）")
            return False
            
    except Exception as e:
        print(f"✗ 点击失败: {e}")
        return False


def activate_window(hwnd):
    """强制激活窗口并置顶"""
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
        
        print("✓ 窗口已激活并置顶")
        return True
    except Exception as e:
        print(f"✗ 激活窗口失败: {e}")
        return False


def paste_fen_workflow(main_hwnd, fen_string):
    """完整的粘贴FEN流程"""
    print("=" * 60)
    print("开始粘贴FEN棋谱")
    print("=" * 60)
    print()
    
    # 1. 复制FEN到剪贴板
    if not set_clipboard(fen_string):
        return False
    
    print()
    
    # 2. 激活窗口
    if not activate_window(main_hwnd):
        return False
    
    time.sleep(0.3)
    
    # 3. 点击"棋局"按钮（相对位置 64, 40）
    print("✓ 点击'棋局'按钮")
    if not click_by_relative_position(main_hwnd, 64, 40):
        return False
    
    # 4. 等待菜单弹出
    time.sleep(0.2)
    
    # 5. 点击"粘贴局面/棋谱"按钮（相对位置 100, 154）
    print("✓ 点击'粘贴局面/棋谱'按钮")
    if not click_by_relative_position(main_hwnd, 100, 154):
        return False
    
    # 6. 尝试查找弹出的菜单控件（调试用）
    print("\n✓ 查找弹出菜单控件（类名 #32768）")
    hwnd = win32gui.FindWindowEx(main_hwnd, 0, '#32768', None)
    
    if hwnd:
        print(f"✓ 找到弹出菜单")
        print(f"  句柄: {hwnd}")
        
        # 获取菜单位置
        menu_rect = win32gui.GetWindowRect(hwnd)
        print(f"  位置: ({menu_rect[0]}, {menu_rect[1]})")
        print(f"  大小: {menu_rect[2] - menu_rect[0]} x {menu_rect[3] - menu_rect[1]}")
    else:
        print("✗ 未找到弹出菜单控件")
    
    print()
    print("=" * 60)
    print("流程完成")
    print("=" * 60)
    
    return True


def debug_mouse_position(main_hwnd):
    """调试模式：显示窗口位置和实时鼠标位置"""
    print("=" * 60)
    print("调试模式：鼠标位置监控")
    print("=" * 60)
    print()
    
    # 激活窗口
    activate_window(main_hwnd)
    time.sleep(0.3)
    
    # 获取窗口位置
    rect = win32gui.GetWindowRect(main_hwnd)
    print(f"主窗口位置信息：")
    print(f"  左上角: ({rect[0]}, {rect[1]})")
    print(f"  右下角: ({rect[2]}, {rect[3]})")
    print(f"  宽度: {rect[2] - rect[0]}")
    print(f"  高度: {rect[3] - rect[1]}")
    print()
    print("=" * 60)
    print("开始监控鼠标位置（按 Ctrl+C 退出）")
    print("请将鼠标移动到'棋局'按钮和'粘贴局面/棋谱'按钮上")
    print("=" * 60)
    print()
    
    try:
        while True:
            # 获取鼠标位置
            x, y = win32api.GetCursorPos()
            
            # 计算相对于窗口的位置
            rel_x = x - rect[0]
            rel_y = y - rect[1]
            
            # 清空当前行并打印
            print(f"\r鼠标位置 - 绝对: ({x}, {y}) | 相对窗口: ({rel_x}, {rel_y})    ", end='', flush=True)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n调试结束")


def find_control_index(parent_hwnd, target_hwnd, class_name):
    """查找目标控件在父控件中的索引位置"""
    index = 0
    current_hwnd = None
    
    while True:
        # 查找下一个同类控件
        current_hwnd = win32gui.FindWindowEx(parent_hwnd, current_hwnd, class_name, None)
        
        if not current_hwnd:
            # 没找到更多控件
            return -1
        
        if current_hwnd == target_hwnd:
            # 找到目标控件
            return index
        
        index += 1


def test_find_engine_button():
    """测试查找"引擎执黑"按钮的位置"""
    print("=" * 60)
    print("测试查找引擎执黑按钮")
    print("=" * 60)
    print()
    
    # 查找主窗口
    main_hwnd = find_main_window()
    if not main_hwnd:
        return
    activate_window(main_hwnd)
    
    print()

    
    # 目标句柄（根据你提供的信息）
    target_controlbar_hwnd = 1970692
    target_toolbar_hwnd = 1710666
    
    # 1. 查找 TControlBar 在主窗口中的位置
    print("步骤1：查找 TControlBar 在主窗口中的索引")
    controlbar_index = find_control_index(main_hwnd, target_controlbar_hwnd, "TControlBar")
    
    if controlbar_index >= 0:
        print(f"✓ TControlBar 是主窗口的第 {controlbar_index} 个 TControlBar 子控件")
        print(f"  句柄: {target_controlbar_hwnd}")
    else:
        print(f"✗ 未找到句柄为 {target_controlbar_hwnd} 的 TControlBar")
        return
    
    print()
    
    # 2. 查找 TToolBar 在 TControlBar 中的位置
    print("步骤2：查找 TToolBar 在 TControlBar 中的索引")
    toolbar_index = find_control_index(target_controlbar_hwnd, target_toolbar_hwnd, "TToolBar")
    
    if toolbar_index >= 0:
        print(f"✓ TToolBar 是 TControlBar 的第 {toolbar_index} 个 TToolBar 子控件")
        print(f"  句柄: {target_toolbar_hwnd}")
    else:
        print(f"✗ 未找到句柄为 {target_toolbar_hwnd} 的 TToolBar")
        return
    
    print()
    print("=" * 60)
    print("定位代码示例")
    print("=" * 60)
    print()
    print("# 通过索引定位控件")
    print(f"# 1. 找到主窗口的第 {controlbar_index} 个 TControlBar")
    print(f"controlbar_hwnd = win32gui.FindWindowEx(main_hwnd, 0, 'TControlBar', None)")
    if controlbar_index > 0:
        print(f"for _ in range({controlbar_index}):")
        print(f"    controlbar_hwnd = win32gui.FindWindowEx(main_hwnd, controlbar_hwnd, 'TControlBar', None)")
    print()
    print(f"# 2. 找到 TControlBar 的第 {toolbar_index} 个 TToolBar")
    print(f"toolbar_hwnd = win32gui.FindWindowEx(controlbar_hwnd, 0, 'TToolBar', None)")
    if toolbar_index > 0:
        print(f"for _ in range({toolbar_index}):")
        print(f"    toolbar_hwnd = win32gui.FindWindowEx(controlbar_hwnd, toolbar_hwnd, 'TToolBar', None)")
    print()
    print("# 3. 点击控件")
    print("win32gui.SendMessage(toolbar_hwnd, win32con.BM_CLICK, 0, 0)")
    print()
    print("=" * 60)


def find_engine_button(main_hwnd):
    """定位"引擎执黑"按钮（TToolBar控件）"""
    try:
        # 1. 找到主窗口的第 0 个 TControlBar
        controlbar_hwnd = win32gui.FindWindowEx(main_hwnd, 0, 'TControlBar', None)
        
        if not controlbar_hwnd:
            print("✗ 未找到 TControlBar")
            return None
        
        print(f"✓ 找到 TControlBar，句柄: {controlbar_hwnd}")
        
        # 2. 找到 TControlBar 的第 1 个 TToolBar
        toolbar_hwnd = win32gui.FindWindowEx(controlbar_hwnd, 0, 'TToolBar', None)
        
        if toolbar_hwnd:
            # 跳过第 0 个，找第 1 个
            toolbar_hwnd = win32gui.FindWindowEx(controlbar_hwnd, toolbar_hwnd, 'TToolBar', None)
        
        if not toolbar_hwnd:
            print("✗ 未找到目标 TToolBar")
            return None
        
        print(f"✓ 找到 TToolBar（引擎执黑按钮），句柄: {toolbar_hwnd}")
        return toolbar_hwnd
        
    except Exception as e:
        print(f"✗ 定位控件失败: {e}")
        return None


def get_button_state(hwnd):
    """获取按钮状态（按下/弹起）"""
    try:
        # 发送 BM_GETSTATE 消息获取按钮状态
        state = win32gui.SendMessage(hwnd, win32con.BM_GETSTATE, 0, 0)
        
        # BST_CHECKED = 0x0001 表示按钮被按下
        is_checked = (state & 0x0001) != 0
        
        print(f"✓ 按钮状态: {'按下' if is_checked else '弹起'}")
        print(f"  状态值: {state}")
        
        return is_checked
        
    except Exception as e:
        print(f"✗ 获取状态失败: {e}")
        return None


def click_button(hwnd):
    """点击按钮"""
    try:
        print("✓ 点击按钮...")
        
        # 发送 BM_CLICK 消息
        win32gui.SendMessage(hwnd, win32con.BM_CLICK, 0, 0)
        
        print("✓ 点击完成")
        return True
        
    except Exception as e:
        print(f"✗ 点击失败: {e}")
        return False


def send_hotkey_ctrl_alt_b(main_hwnd):
    """发送快捷键 Ctrl+Alt+B"""
    try:
        # 激活窗口
        activate_window(main_hwnd)
        time.sleep(0.3)
        
        print("✓ 发送快捷键 Ctrl+Alt+B")
        
        # 按下 Ctrl、Alt、B
        VK_CONTROL = 0x11
        VK_MENU = 0x12  # Alt键
        VK_B = 0x42
        
        win32api.keybd_event(VK_CONTROL, 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(VK_MENU, 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(VK_B, 0, 0, 0)
        time.sleep(0.05)
        
        # 释放按键（顺序相反）
        win32api.keybd_event(VK_B, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.05)
        win32api.keybd_event(VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.05)
        win32api.keybd_event(VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        
        print("✓ 快捷键发送完成")
        return True
        
    except Exception as e:
        print(f"✗ 发送快捷键失败: {e}")
        return False


def test_engine_button():
    """测试"引擎执黑"按钮（使用快捷键 Ctrl+Alt+B）"""
    print("=" * 60)
    print("测试引擎执黑按钮 - 使用快捷键")
    print("=" * 60)
    print()
    
    # 查找主窗口
    main_hwnd = find_main_window()
    if not main_hwnd:
        return
    
    print()
    
    # 发送快捷键
    print("步骤1：发送快捷键 Ctrl+Alt+B")
    if not send_hotkey_ctrl_alt_b(main_hwnd):
        return
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("请观察鲨鱼象棋窗口，看看'引擎执黑'按钮是否被触发")
    print("=" * 60)


def main():
    print("=" * 60)
    print("鲨鱼象棋 - FEN棋谱粘贴测试")
    print("=" * 60)
    print()
    
    # 测试用的FEN棋谱（开局局面）
    test_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    
    print(f"测试FEN: {test_fen}")
    print()
    
    # 查找主窗口
    main_hwnd = find_main_window()
    if not main_hwnd:
        return
    
    print()
    
    # 测试引擎按钮
    test_engine_button()
    
    # # 测试查找引擎按钮索引
    # test_find_engine_button()
    
    # # 执行粘贴流程
    # paste_fen_workflow(main_hwnd, test_fen)
    
    # # 调试模式（显示鼠标位置）
    # debug_mouse_position(main_hwnd)


if __name__ == "__main__":
    # 自动请求管理员权限
    # run_as_admin()
    
    # 执行主程序
    main()
