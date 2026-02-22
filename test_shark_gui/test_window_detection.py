"""
测试脚本：检测鲨鱼象棋程序的GUI窗口
用于枚举所有Windows顶层窗口，找到鲨鱼象棋程序
"""

import win32gui
import win32process
import psutil


def get_process_name(hwnd):
    """获取窗口对应的进程名称"""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        return process.name()
    except:
        return "Unknown"


def enum_windows_callback(hwnd, windows_list):
    """窗口枚举回调函数"""
    if win32gui.IsWindowVisible(hwnd):
        window_title = win32gui.GetWindowText(hwnd)
        if window_title:  # 只显示有标题的窗口
            class_name = win32gui.GetClassName(hwnd)
            process_name = get_process_name(hwnd)
            windows_list.append({
                'hwnd': hwnd,
                'title': window_title,
                'class': class_name,
                'process': process_name
            })


def list_all_windows():
    """列出所有顶层窗口"""
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows


def find_shark_chess_window(windows):
    """查找鲨鱼象棋程序窗口"""
    # 可能的关键词
    keywords = ['鲨鱼', 'shark', '象棋', 'chess', 'xiangqi']
    
    matches = []
    for window in windows:
        title_lower = window['title'].lower()
        process_lower = window['process'].lower()
        
        for keyword in keywords:
            if keyword in title_lower or keyword in process_lower:
                matches.append(window)
                break
    
    return matches


def main():
    print("=" * 80)
    print("开始枚举Windows顶层窗口...")
    print("=" * 80)
    print()
    
    # 获取所有窗口
    windows = list_all_windows()
    
    print(f"找到 {len(windows)} 个可见窗口\n")
    
    # 显示所有窗口
    print("所有顶层窗口列表：")
    print("-" * 80)
    for i, window in enumerate(windows, 1):
        print(f"{i}. 窗口句柄: {window['hwnd']}")
        print(f"   标题: {window['title']}")
        print(f"   类名: {window['class']}")
        print(f"   进程: {window['process']}")
        print()
    
    # 查找鲨鱼象棋窗口
    print("=" * 80)
    print("查找鲨鱼象棋程序窗口...")
    print("=" * 80)
    print()
    
    matches = find_shark_chess_window(windows)
    
    if matches:
        print(f"找到 {len(matches)} 个可能的鲨鱼象棋窗口：")
        print("-" * 80)
        for i, window in enumerate(matches, 1):
            print(f"{i}. 窗口句柄: {window['hwnd']}")
            print(f"   标题: {window['title']}")
            print(f"   类名: {window['class']}")
            print(f"   进程: {window['process']}")
            print()
    else:
        print("未找到鲨鱼象棋程序窗口")
        print("请确保鲨鱼象棋程序正在运行")
        print()
        print("提示：请在上面的窗口列表中手动查找鲨鱼象棋程序")


if __name__ == "__main__":
    main()

