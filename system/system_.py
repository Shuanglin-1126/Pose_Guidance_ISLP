import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from datetime import datetime
import subprocess
import sys
from pathlib import Path

# 外部脚本路径
SCRIPT_PATHS = {
    "isolated": r"\project\sapiens\slr\tools\sample.py",
    "continuous": r"\project\AdaptSign-main\sample.py",
    "generate": r"\project\RAE\src\sample_system.py"
}
# 你的可执行脚本文件路径（孤立手语识别、连续手语识别、孤立手语生成）
ROOT_DIR = {
    "isolated": r"\project\sapiens",
    "continuous": r"\project\AdaptSign-main",
    "generate": r"\project\RAE"
}
# 你的模型权重存放路径（孤立手语识别、连续手语识别、孤立手语生成）

class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手语双向翻译系统")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.current_module = None
        self.log_file_path = f"./system_logs/system_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # ========== 顶部功能选择栏 ==========
        self.top_frame = ttk.LabelFrame(self, text="功能模块选择")
        self.top_frame.pack(fill=tk.X, padx=20, pady=10)

        self.btn_isolated = ttk.Button(
            self.top_frame, text="孤立手语识别",
            command=lambda: self.switch_module("isolated")
        )
        self.btn_isolated.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_continuous = ttk.Button(
            self.top_frame, text="连续手语识别",
            command=lambda: self.switch_module("continuous")
        )
        self.btn_continuous.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_generate = ttk.Button(
            self.top_frame, text="文本转手语生成",
            command=lambda: self.switch_module("generate")
        )
        self.btn_generate.pack(side=tk.LEFT, padx=20, pady=10)

        # ========== 功能模块容器 ==========
        self.module_container = ttk.Frame(self)
        self.module_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 初始化三个功能模块
        self.isolated_module = IsolatedSignModule(self.module_container, self)
        self.continuous_module = ContinuousSignModule(self.module_container, self)
        self.generate_module = SignGenerateModule(self.module_container, self)

        # 默认显示孤立手语识别模块
        self.switch_module("isolated")

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def switch_module(self, module_name):
        """切换功能模块（切换时清空视频画面）"""
        if self.current_module:
            self.current_module.stop_all(reset_frame=True)

        # 隐藏所有模块
        self.isolated_module.pack_forget()
        self.continuous_module.pack_forget()
        self.generate_module.pack_forget()

        # 显示选中的模块
        if module_name == "isolated":
            self.current_module = self.isolated_module
            self.isolated_module.pack(fill=tk.BOTH, expand=True)
        elif module_name == "continuous":
            self.current_module = self.continuous_module
            self.continuous_module.pack(fill=tk.BOTH, expand=True)
        elif module_name == "generate":
            self.current_module = self.generate_module
            self.generate_module.pack(fill=tk.BOTH, expand=True)

        self.log_message("模块切换", "成功", f"已切换至{self.current_module.module_name}模块")

    def log_message(self, operation, status, detail):
        """日志记录：界面+本地文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {operation} | {status} | {detail}\n"

        # 写入本地日志
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"日志写入失败: {e}")

        # 显示到当前模块的日志框
        if self.current_module and hasattr(self.current_module, "log_text"):
            self.current_module.log_text.config(state="normal")
            if status == "成功":
                self.current_module.log_text.insert(tk.END, log_line, "success")
            elif status == "异常":
                self.current_module.log_text.insert(tk.END, log_line, "error")
            elif status == "进行中":
                self.current_module.log_text.insert(tk.END, log_line, "processing")
            else:
                self.current_module.log_text.insert(tk.END, log_line)
            self.current_module.log_text.see(tk.END)
            self.current_module.log_text.config(state="disabled")

    def on_closing(self):
        """关闭时释放资源+保存日志"""
        if self.current_module:
            self.current_module.stop_all(reset_frame=True)
        self.log_message("系统关闭", "成功", "系统正常退出，资源已释放")
        self.destroy()


class BaseSignModule(ttk.Frame):
    """基础模块类"""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.cap = None
        self.stop_event = threading.Event()
        self.video_label = None
        self.default_image = self._load_default_image()
        self.last_frame = None
        self.current_video_path = None
        # 新增：存放文件夹内所有图片路径
        self.image_seq = []
        # 播放帧率
        self.fps = 30

    def _load_default_image(self):
        """加载默认占位图"""
        try:
            img = Image.open("default.png")
        except:
            img = Image.new('RGB', (640, 480), color='lightgray')
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def create_video_display(self, parent):
        """创建视频预览区"""
        video_frame = ttk.LabelFrame(parent, text="视频预览")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = ttk.Label(video_frame, image=self.default_image)
        self.video_label.pack()
        return video_frame

    def select_folder(self):
        """选择帧文件夹：读取所有图片并播放序列"""
        folder_path = filedialog.askdirectory(title="选择帧文件夹")
        if not folder_path:
            return

        self.stop_all()
        self.current_video_path = folder_path
        self.app.log_message("选择目录", "成功", f"已选中文件夹：{folder_path}")

        # 1. 读取文件夹中所有图片，按名称排序
        self.image_seq = []
        suffix = ('.jpg', '.jpeg', '.png', '.bmp')
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith(suffix):
                self.image_seq.append(os.path.join(folder_path, name))

        if not self.image_seq:
            self.app.log_message("提示", "警告", "文件夹内未找到图片文件")
            return

        # 2. 启动图片序列播放线程
        self.stop_event.clear()
        threading.Thread(target=self._play_image_sequence, daemon=True).start()

    def _play_image_sequence(self):
        """播放图片序列（模拟视频），循环播放"""
        delay = 1.0 / self.fps
        while not self.stop_event.is_set():
            for img_path in self.image_seq:
                if self.stop_event.is_set():
                    return
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # 转RGB + 缩放
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                img = Image.fromarray(frame_rgb)
                self.last_frame = ImageTk.PhotoImage(image=img)

                # 更新UI（主线程执行）
                self.after(0, lambda f=self.last_frame: self.video_label.config(image=f))
                self.video_label.image = self.last_frame
                time.sleep(delay)

    def open_camera(self):
        """打开摄像头"""
        self.stop_all()
        self.current_video_path = "camera"
        self.image_seq.clear()  # 清空图片序列
        self.cap = cv2.VideoCapture(0)
        self.stop_event.clear()
        threading.Thread(target=self._play_video, daemon=True).start()
        self.app.log_message("打开摄像头", "成功", "已打开摄像头")

    def _play_video(self):
        """播放摄像头视频流"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.last_frame:
                    self.after(0, lambda: self.video_label.config(image=self.last_frame))
                    self.video_label.image = self.last_frame
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            img = Image.fromarray(frame_rgb)
            self.last_frame = ImageTk.PhotoImage(image=img)

            self.after(0, lambda: self.video_label.config(image=self.last_frame))
            self.video_label.image = self.last_frame
            time.sleep(1 / self.fps)

    def stop_all(self, reset_frame=True):
        """停止所有播放"""
        self.stop_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None

        if reset_frame:
            self.last_frame = None
            self.current_video_path = None
            self.image_seq.clear()
            self.after(0, lambda: self.video_label.config(image=self.default_image))
            self.video_label.image = self.default_image


# ==================== 孤立手语识别模块（和截图布局一致） ====================
class IsolatedSignModule(BaseSignModule):
    module_name = "孤立手语识别"

    def __init__(self, parent, app):
        super().__init__(parent, app)

        # 左侧视频区域
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_video_display(left_frame)

        # 视频控制按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="选择本地视频/文件夹", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="打开摄像头", command=self.open_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止播放", command=lambda: self.stop_all(reset_frame=False)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="开始识别", command=self.run_external_script).pack(side=tk.LEFT, padx=5)

        # 右侧结果+日志区域（和截图完全一致）
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 结果展示区
        result_frame = ttk.LabelFrame(right_frame, text="孤立手语识别结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_text = tk.Text(result_frame, height=10, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 系统日志区
        log_frame = ttk.LabelFrame(right_frame, text="系统日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=15, width=40, state="disabled", font=("微软雅黑", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("processing", foreground="blue")

    def run_external_script(self):
        """运行外部脚本，并把结果写到结果区"""
        if not self.current_video_path:
            messagebox.showwarning("警告", "请先选择帧文件夹或打开摄像头！")
            return
        self.app.log_message("开始识别", "进行中", "正在调用外部脚本...")

        script_path = SCRIPT_PATHS["isolated"]
        work_dir = ROOT_DIR["isolated"]

        def task():
            inline_code = (
                f"import sys, os; "
                f"os.chdir(r'{work_dir}'); "
                f"sys.path.insert(0, r'{work_dir}'); "
                f"import builtins; "
                f"builtins.__dict__['__file__'] = r'{script_path}'; "
                f"exec(open(r'{script_path}', encoding='utf-8').read())"
            )

            cmd = [
                sys.executable,
                "-c",  # 执行后面的字符串代码
                inline_code,
                "--input",
                self.current_video_path
            ]
            # 运行脚本并捕获输出
            result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
            if result.returncode == 0:
                self.app.log_message("识别完成", "成功", "连续手语识别脚本运行结束")
            else:
                self.app.log_message("识别失败", "异常", f"脚本返回错误码: {result.returncode}")

            data_iso = {}
            try:
                with open(r'\project\RAE\system\ISLR.txt', "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("预测类别"):
                        data_iso['pred'] = line.split("：")[-1].strip()
                    elif line.startswith("置信度"):
                        data_iso['conf'] = line.split("：")[-1].split("(")[0].strip()
                        data_iso['conf_pct'] = line.split("(")[-1].rstrip(")").strip()
                    elif line.startswith("推理耗时"):
                        data_iso['time'] = line.split("：")[-1].strip()
                isolated_result = """孤立手语识别结果：
                        - {}
                        - 置信度：{}
                        - {}
                        - 视频源：{}""".format(data_iso['pred'], data_iso['conf_pct'], data_iso['time'],
                                              "摄像头" if self.current_video_path == "camera" else os.path.basename(
                                                  self.current_video_path))

                self.result_text.insert(tk.END, isolated_result)

            except Exception as e:
                self.app.log_message(f"读取失败: {e}")

        threading.Thread(target=task, daemon=True).start()






# ==================== 连续手语识别模块（同布局） ====================
class ContinuousSignModule(BaseSignModule):
    module_name = "连续手语识别"

    def __init__(self, parent, app):
        super().__init__(parent, app)

        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_video_display(left_frame)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="选择本地视频/文件夹", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="打开摄像头", command=self.open_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止播放", command=lambda: self.stop_all(reset_frame=False)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="开始连续识别", command=self.run_external).pack(side=tk.LEFT, padx=5)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        result_frame = ttk.LabelFrame(right_frame, text="连续手语识别结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_text = tk.Text(result_frame, height=10, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        log_frame = ttk.LabelFrame(right_frame, text="系统日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=15, width=40, state="disabled", font=("微软雅黑", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("processing", foreground="blue")

    def run_external(self):
        if not self.current_video_path:
            messagebox.showwarning("警告", "请先选择帧文件夹或打开摄像头！")
            return

        script_path = SCRIPT_PATHS["continuous"]
        work_dir = ROOT_DIR["continuous"]
        self.app.log_message("开始连续识别", "进行中", "正在调用外部脚本...")

        def task():
            inline_code = (
                f"import sys, os; "
                f"os.chdir(r'{work_dir}'); "
                f"sys.path.insert(0, r'{work_dir}'); "
                f"import builtins; "
                f"builtins.__dict__['__file__'] = r'{script_path}'; "
                f"exec(open(r'{script_path}', encoding='utf-8').read())"
            )

            cmd = [
                sys.executable,
                "-c",  # 执行后面的字符串代码
                inline_code,
                "--input",
                self.current_video_path
            ]
            result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

            if result.returncode == 0:
                self.app.log_message("识别完成", "成功", "连续手语识别脚本运行结束")
            else:
                self.app.log_message("识别失败", "异常", f"脚本返回错误码: {result.returncode}")

            data_con = {}
            try:
                with open(r'\project\RAE\system\CSLR.txt', "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("预测句子"):
                        data_con['pred'] = line.split("：")[-1].strip()
                    elif line.startswith("推理耗时"):
                        data_con['time'] = line.split("：")[-1].strip()
                con_result = """连续手语识别结果：
                        - {}
                        - {}
                        - 视频源：{}""".format(data_con['pred'], data_con['time'],
                                              "摄像头" if self.current_video_path == "camera" else os.path.basename(
                                                  self.current_video_path))

                self.result_text.insert(tk.END, con_result)

            except Exception as e:
                self.app.log_message(f"读取失败: {e}")

        threading.Thread(target=task, daemon=True).start()



# ==================== 文本转手语生成模块 ====================
class SignGenerateModule(BaseSignModule):
    module_name = "文本转手语生成"

    def __init__(self, parent, app):
        super().__init__(parent, app)

        # 新增：生成结果帧存放目录（根据你实际路径修改）
        self.gen_frame_dir = r"F:\SLRdataset\WLASL\gen_frames"

        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(left_frame, text="输入待生成的文字")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.input_text = tk.Text(input_frame, height=10, width=40)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.insert(tk.END, "请输入要生成手语视频的文字内容...")

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="清空输入", command=self.clear_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="生成手语视频", command=self.run_external).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止播放", command=lambda: self.stop_all(reset_frame=False)).pack(side=tk.LEFT, padx=5)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_video_display(right_frame)

        result_frame = ttk.LabelFrame(right_frame, text="生成信息")
        result_frame.pack(fill=tk.X, padx=10, pady=10)
        self.result_label = ttk.Label(result_frame, text="未生成视频，输入文字后点击「生成手语视频」")
        self.result_label.pack(padx=5, pady=5)

        log_frame = ttk.LabelFrame(right_frame, text="系统日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=10, width=40, state="disabled", font=("微软雅黑", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("processing", foreground="blue")

    def clear_input(self):
        self.input_text.delete(1.0, tk.END)
        self.result_label.config(text="未生成视频，输入文字后点击「生成手语视频」")

    def _play_generated_frames(self, folder_path):
        """加载生成的帧图片并播放（复用基类图片播放逻辑）"""
        if not os.path.isdir(folder_path):
            self.app.log_message("播放", "警告", f"帧目录不存在：{folder_path}")
            return

        self.stop_all(reset_frame=False)
        self.current_video_path = folder_path

        # 读取排序图片
        self.image_seq = []
        suffix = ('.jpg', '.jpeg', '.png', '.bmp')
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith(suffix):
                self.image_seq.append(os.path.join(folder_path, name))

        if not self.image_seq:
            self.app.log_message("播放", "警告", "生成目录下未找到图片帧")
            return

        # 启动播放线程
        self.stop_event.clear()
        self.fps = 5
        threading.Thread(target=self._play_image_sequence, daemon=True).start()
        self.app.log_message("播放", "成功", "开始播放生成的手语帧序列")

    def run_external(self):
        # 获取输入文本
        text_content = self.input_text.get("1.0", tk.END).strip()
        if not text_content or text_content == "请输入要生成手语视频的文字内容...":
            messagebox.showwarning("警告", "请输入有效的文字内容！")
            return

        script_path = SCRIPT_PATHS["generate"]
        work_dir = ROOT_DIR["generate"]
        self.app.log_message("生成视频", "进行中", f"正在为「{text_content}」生成手语视频...")

        def task():
            inline_code = (
                f"import sys, os; "
                f"os.chdir(r'{work_dir}'); "
                f"sys.path.insert(0, r'{work_dir}'); "
                f"import builtins; "
                f"builtins.__dict__['__file__'] = r'{script_path}'; "
                f"exec(open(r'{script_path}', encoding='utf-8').read())"
            )

            # 核心改动：传两个参数 --text 文本内容  --out-dir 输出帧目录
            cmd = [
                sys.executable,
                "-c",
                inline_code,
                "--input", text_content,
            ]
            # 指定编码防止乱码
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            if result.returncode == 0:
                self.result_label.config(text=f"生成成功: {text_content}")
                self.app.log_message("生成视频", "成功", "手语生成脚本运行结束")
                # 生成完成后，自动加载并播放图片帧
                self._play_generated_frames(r'\project\RAE\system\sample_result')
            else:
                err_msg = result.stderr[:300]  # 截断避免过长
                self.result_label.config(text=f"生成失败: {err_msg}")
                self.app.log_message("生成视频", "异常", f"错误码: {result.returncode}, 详情: {err_msg}")

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()