"""
测试脚本：遍历鲨鱼象棋窗口的控件树
用于查看窗口内所有子控件的层级结构
"""

import win32gui
import win32con
import win32api


def get_window_rect(hwnd):
    """获取窗口位置和大小"""
    try:
        rect = win32gui.GetWindowRect(hwnd)
        return {
            'left': rect[0],
            'top': rect[1],
            'right': rect[2],
            'bottom': rect[3],
            'width': rect[2] - rect[0],
            'height': rect[3] - rect[1]
        }
    except:
        return None


def is_window_visible(hwnd):
    """检查窗口是否可见"""
    try:
        return win32gui.IsWindowVisible(hwnd)
    except:
        return False


def enum_child_windows(parent_hwnd, level=0):
    """递归枚举子窗口"""
    controls = []
    
    def callback(hwnd, param):
        try:
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = get_window_rect(hwnd)
            visible = is_window_visible(hwnd)
            
            control_info = {
                'hwnd': hwnd,
                'level': level,
                'title': title,
                'class': class_name,
                'rect': rect,
                'visible': visible
            }
            
            controls.append(control_info)
            
            # 递归获取子控件
            child_controls = enum_child_windows(hwnd, level + 1)
            controls.extend(child_controls)
            
        except Exception as e:
            print(f"错误: {e}")
        
        return True
    
    try:
        win32gui.EnumChildWindows(parent_hwnd, callback, None)
    except:
        pass
    
    return controls


def print_control_tree(controls):
    """打印控件树"""
    for control in controls:
        indent = "  " * control['level']
        
        # 基本信息
        print(f"{indent}├─ 句柄: {control['hwnd']}")
        print(f"{indent}│  类名: {control['class']}")
        
        if control['title']:
            print(f"{indent}│  标题: {control['title']}")
        
        print(f"{indent}│  可见: {'是' if control['visible'] else '否'}")
        
        if control['rect']:
            rect = control['rect']
            print(f"{indent}│  位置: ({rect['left']}, {rect['top']})")
            print(f"{indent}│  大小: {rect['width']} x {rect['height']}")
        
        print(f"{indent}│")


def find_window_by_title(title_keyword):
    """根据标题关键词查找窗口"""
    result = {'hwnd': None}
    
    def callback(hwnd, param):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title_keyword in window_title:
                result['hwnd'] = hwnd
                return False  # 停止枚举
        return True
    
    win32gui.EnumWindows(callback, None)
    return result['hwnd']


def main():
    print("=" * 80)
    print("鲨鱼象棋控件树分析")
    print("=" * 80)
    print()
    
    # 方法1: 使用已知的窗口句柄
    shark_hwnd = 1839672  # TForm1 窗口句柄
    
    # 方法2: 动态查找窗口（推荐）
    # shark_hwnd = find_window_by_title("鲨鱼象棋 V")
    
    if not shark_hwnd:
        print("错误: 未找到鲨鱼象棋窗口")
        print("请确保鲨鱼象棋程序正在运行")
        return
    
    # 验证窗口是否存在
    try:
        window_title = win32gui.GetWindowText(shark_hwnd)
        window_class = win32gui.GetClassName(shark_hwnd)
        print(f"目标窗口: {window_title}")
        print(f"窗口类名: {window_class}")
        print(f"窗口句柄: {shark_hwnd}")
        print()
    except:
        print(f"错误: 窗口句柄 {shark_hwnd} 无效")
        return
    
    # 获取窗口大小
    rect = get_window_rect(shark_hwnd)
    if rect:
        print(f"窗口位置: ({rect['left']}, {rect['top']})")
        print(f"窗口大小: {rect['width']} x {rect['height']}")
        print()
    
    print("=" * 80)
    print("开始遍历控件树...")
    print("=" * 80)
    print()
    
    # 枚举所有子控件
    controls = enum_child_windows(shark_hwnd)
    
    print(f"找到 {len(controls)} 个子控件\n")
    print("控件树结构：")
    print("-" * 80)
    
    # 打印控件树
    print_control_tree(controls)
    
    print("-" * 80)
    print()
    
    # 统计信息
    print("统计信息：")
    print(f"  总控件数: {len(controls)}")
    
    visible_count = sum(1 for c in controls if c['visible'])
    print(f"  可见控件: {visible_count}")
    
    # 按类名分组
    class_counts = {}
    for control in controls:
        class_name = control['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n控件类型分布：")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()

