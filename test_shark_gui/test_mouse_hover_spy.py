"""
鼠标悬停控件识别工具
实时显示鼠标下方的控件信息，方便RPA开发时定位目标控件
按 Ctrl+C 退出
"""

import win32gui
import win32api
import win32con
import time
import sys
import os


def get_window_info(hwnd):
    """获取窗口详细信息"""
    try:
        title = win32gui.GetWindowText(hwnd)
        class_name = win32gui.GetClassName(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        
        return {
            'hwnd': hwnd,
            'title': title,
            'class': class_name,
            'rect': rect,
            'width': rect[2] - rect[0],
            'height': rect[3] - rect[1]
        }
    except:
        return None


def get_parent_chain(hwnd):
    """获取控件的父级链"""
    chain = []
    current = hwnd
    
    while current:
        info = get_window_info(current)
        if info:
            chain.append(info)
        
        try:
            parent = win32gui.GetParent(current)
            if parent == 0 or parent == current:
                break
            current = parent
        except:
            break
    
    return chain


def is_shark_chess_window(hwnd):
    """判断是否是鲨鱼象棋窗口或其子控件"""
    chain = get_parent_chain(hwnd)
    for info in chain:
        if '鲨鱼象棋' in info['title']:
            return True
    return False

def is_not_pycharm_window(hwnd):
    """判断不是pycharm输出台"""
    chain = get_parent_chain(hwnd)
    for info in chain:
        if 'AIchess' in info['title']:
            return False
    return True


def print_control_info(info, parent_chain):
    """打印控件信息"""
    os.system('cls')
    
    print("=" * 80)
    print("【当前控件】")
    print(f"句柄: {info['hwnd']}")
    print(f"类名: {info['class']}")
    print(f"标题: {info['title']}" if info['title'] else "标题: (无)")
    print(f"位置: ({info['rect'][0]}, {info['rect'][1]})")
    print(f"大小: {info['width']} x {info['height']}")
    
    if len(parent_chain) > 1:
        print("-" * 80)
        print("【父级控件链】")
        for i, parent in enumerate(parent_chain[1:], 1):
            indent = "  " * i
            print(f"{indent}↑ 句柄: {parent['hwnd']}")
            print(f"{indent}  类名: {parent['class']}")
            if parent['title']:
                print(f"{indent}  标题: {parent['title']}")
    
    print("-" * 80)
    print("【RPA代码示例】")
    print(f"# 通过句柄定位控件")
    print(f"hwnd = {info['hwnd']}")
    print(f"# 或通过类名和标题查找")
    if info['title']:
        print(f"# hwnd = win32gui.FindWindowEx(parent_hwnd, 0, '{info['class']}', '{info['title']}')")
    else:
        print(f"# hwnd = win32gui.FindWindowEx(parent_hwnd, 0, '{info['class']}', None)")
    print()
    print("# 点击控件")
    print("win32gui.SendMessage(hwnd, win32con.BM_CLICK, 0, 0)")
    print()
    print("# 获取文本")
    print("text = win32gui.GetWindowText(hwnd)")
    print()
    print("# 设置文本")
    print("win32gui.SendMessage(hwnd, win32con.WM_SETTEXT, 0, '新文本')")
    print("=" * 80)
    print("提示：将鼠标移动到鲨鱼象棋窗口的控件上查看信息 | 按 Ctrl+C 退出")


def main():
    print("=" * 80)
    print("鼠标悬停控件识别工具")
    print("=" * 80)
    print()
    print("使用说明：")
    print("1. 将鼠标移动到鲨鱼象棋窗口的任意控件上")
    print("2. 工具会实时显示该控件的详细信息")
    print("3. 按 Ctrl+C 退出程序")
    print()
    print("开始监控...")
    print("-" * 80)
    
    last_hwnd = None
    
    try:
        while True:
            # 获取鼠标位置
            x, y = win32api.GetCursorPos()
            
            # 获取鼠标下的窗口句柄
            hwnd = win32gui.WindowFromPoint((x, y))
            
            # 如果句柄变化且是鲨鱼象棋窗口
            if hwnd != last_hwnd and hwnd != 0:
                if is_not_pycharm_window(hwnd):
                    info = get_window_info(hwnd)
                    if info:
                        parent_chain = get_parent_chain(hwnd)
                        print_control_info(info, parent_chain)
                        last_hwnd = hwnd
            
            # 短暂延迟，避免CPU占用过高
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)


if __name__ == "__main__":
    main()

