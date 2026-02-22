import tkinter as tk
import threading
import ctypes

# 强制 Windows 识别真实物理像素，防止 UI 缩放导致后续截屏和坐标错位
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class ChessMVP:
    def __init__(self, root):
        self.root = root
        self.root.title("单兵作战 - 极简版")
        self.root.attributes("-topmost", True)  # 必须置顶
        self.root.geometry("280x160")

        # 核心状态位
        self.ai_enabled = False

        # --- UI 绘制 ---
        self.status_label = tk.Label(root, text="状态: 等待初始化", fg="gray", font=("微软雅黑", 10, "bold"))
        self.status_label.pack(pady=10)

        # 按钮 1：初始化与框选
        self.btn_init = tk.Button(root, text="1. 框选棋盘 & 链接鲨鱼", command=self.on_init_click, height=2)
        self.btn_init.pack(fill=tk.X, padx=20, pady=5)

        # 按钮 2：开启/关闭 AI 辅助
        self.btn_ai = tk.Button(root, text="2. 开启 AI 辅助 (单次触发)", command=self.on_toggle_ai, height=2,
                                state=tk.DISABLED)
        self.btn_ai.pack(fill=tk.X, padx=20, pady=5)

    def update_status(self, text, color="black"):
        """线程安全的 UI 更新方法"""
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    # ==================== 按钮 1 逻辑 ====================
    def on_init_click(self):
        """点击初始化按钮，开启独立线程防止界面卡死"""
        self.btn_init.config(state=tk.DISABLED)
        self.update_status("正在执行初始化...", "orange")
        threading.Thread(target=self.init_task, daemon=True).start()

    def init_task(self):
        """后台初始化任务"""
        try:
            # 导入模块
            from vision import initialize_vision
            from shark_control import initialize_shark_control
            
            # 1. 初始化视觉模块
            print("\n" + "="*60)
            print("任务1：初始化视觉模块")
            print("="*60)
            success = initialize_vision()
            
            if not success:
                self.update_status("视觉初始化失败", "red")
                self.root.after(0, lambda: self.btn_init.config(state=tk.NORMAL))
                return
            
            # 2. 初始化鲨鱼象棋控制
            print("\n" + "="*60)
            print("任务2：初始化鲨鱼象棋控制")
            print("="*60)
            success = initialize_shark_control()
            
            if not success:
                self.update_status("鲨鱼象棋连接失败", "red")
                self.root.after(0, lambda: self.btn_init.config(state=tk.NORMAL))
                return
            
            # 初始化完成后，解锁第二个按钮
            self.root.after(0, lambda: self.btn_ai.config(state=tk.NORMAL))
            self.update_status("初始化完成！随时可开启AI", "green")
            
        except Exception as e:
            self.update_status(f"初始化异常: {str(e)}", "red")
            self.root.after(0, lambda: self.btn_init.config(state=tk.NORMAL))
            print(f"❌ 初始化异常: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 按钮 2 逻辑 ====================
    def on_toggle_ai(self):
        """点击开启/关闭 AI，切换状态并开启工作线程"""
        if not self.ai_enabled:
            # 准备开启 AI
            self.ai_enabled = True
            self.btn_ai.config(text="关闭 AI 辅助", fg="red")
            self.update_status("AI 处理中，请勿动鼠标...", "red")

            # 开启单次执行线程，防止阻塞 UI
            threading.Thread(target=self.start_ai_task, daemon=True).start()
        else:
            # 准备关闭 AI
            self.ai_enabled = False
            self.btn_ai.config(text="2. 开启 AI 辅助 (单次)", fg="black")
            self.update_status("AI 已关闭", "gray")

            threading.Thread(target=self.stop_ai_task, daemon=True).start()

    def start_ai_task(self):
        """后台开启 AI 任务：感知 -> 生成 -> UI 操控"""
        try:
            # 导入模块
            from vision import capture_and_recognize
            from shark_control import send_fen_and_start_engine
            
            # 1. 截取棋盘并识别为FEN码
            print("\n" + "="*60)
            print("步骤1：识别棋盘")
            print("="*60)
            fen_string = capture_and_recognize()
            
            if fen_string is None:
                self.update_status("识别失败，请重试", "red")
                return
            
            # 2. 发送FEN码并开启引擎
            print("\n" + "="*60)
            print("步骤2：发送FEN码并开启引擎")
            print("="*60)
            success = send_fen_and_start_engine(fen_string)
            
            if not success:
                self.update_status("发送FEN失败", "red")
                return
            
            self.update_status("FEN 已发送，皮卡运算中", "blue")
            
        except Exception as e:
            self.update_status(f"AI任务异常: {str(e)}", "red")
            print(f"❌ AI任务异常: {e}")
            import traceback
            traceback.print_exc()

    def stop_ai_task(self):
        """后台关闭 AI 任务"""
        try:
            # 导入模块
            from shark_control import stop_all_engines
            
            # 关闭所有引擎
            print("\n" + "="*60)
            print("关闭引擎")
            print("="*60)
            success = stop_all_engines()
            
            if success:
                self.update_status("已停止，等待下一次开启", "green")
            else:
                self.update_status("关闭引擎失败", "red")
                
        except Exception as e:
            self.update_status(f"关闭异常: {str(e)}", "red")
            print(f"❌ 关闭异常: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChessMVP(root)
    # 放置在屏幕右侧边缘，避免遮挡棋盘
    root.geometry("+1400+280")
    root.mainloop()
