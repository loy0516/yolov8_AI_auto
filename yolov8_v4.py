import threading
import cv2
import mss
import ctypes
import pyautogui
import time
import numpy as np
import queue
import customtkinter as ctk
import os
from ultralytics import YOLO
from pynput.mouse import Controller


# **自动获取当前路径下的所有 .pt 模型**
def get_model_list():
    model_files = [f for f in os.listdir() if f.endswith(".pt") or f.endswith(".onnx")]
    return model_files if model_files else ["选择模型"]


# 定义 Windows 数据结构
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT_U)]


# UI 界面类
class AI_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Auto UI")
        self.master.geometry("400x350")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 运行状态
        self.running = False

        # 运行状态标签
        self.status_label = ctk.CTkLabel(master, text="状态: 未运行", fg_color="red")
        self.status_label.pack(pady=5)

        # **模型选择下拉框**
        self.model_var = ctk.StringVar(value="选择模型")

        self.model_dropdown = ctk.CTkComboBox(
            master, values=get_model_list(), variable=self.model_var
        )
        self.model_dropdown.pack(pady=5)

        # **预览开关**
        self.preview_var = ctk.BooleanVar(value=False)
        self.preview_checkbox = ctk.CTkCheckBox(
            master, text="启用实时预览", variable=self.preview_var, command=self.toggle_preview
        )
        self.preview_checkbox.pack(pady=5)

        # 启动按钮
        self.start_button = ctk.CTkButton(master, text="启动", command=self.start_ai)
        self.start_button.pack(pady=10)

        # 停止按钮
        self.stop_button = ctk.CTkButton(master, text="停止", command=self.stop_ai)
        self.stop_button.pack(pady=10)

        # **刷新模型列表按钮**
        self.refresh_button = ctk.CTkButton(master, text="刷新模型列表", command=self.refresh_model_list)
        self.refresh_button.pack(pady=10)

        # 创建 AI_Auto 实例
        self.ai_auto = AI_Auto(self)

    def refresh_model_list(self):
        """ 刷新模型下拉框 """
        new_model_list = get_model_list()
        self.model_dropdown.configure(values=new_model_list)
        if new_model_list:
            self.model_var.set(new_model_list[0])

    def toggle_preview(self):
        """ 切换预览状态 """
        self.ai_auto.toggle_preview()

    def start_ai(self):
        if not self.running:
            self.running = True
            self.status_label.configure(text="状态: 运行中", fg_color="green")
            self.ai_auto.load_model(self.model_var.get())
            threading.Thread(target=self.ai_auto.main, daemon=True).start()

    def stop_ai(self):
        if self.running:
            self.running = False
            self.status_label.configure(text="状态: 停止", fg_color="red")
            self.ai_auto.spot = True  # 停止检测
            # if self.ai_auto.preview_enabled:   # 取消勾选
            #     self.ai_auto.preview_enabled = False
            #     self.preview_var.set(False)


# AI 逻辑类
class AI_Auto:
    def __init__(self, ui):
        self.ui = ui
        self.spot = False
        self.model = None  # 初始不加载模型
        self.mouse = Controller()
        self.queue = queue.Queue()
        self.preview_enabled = False  # 是否启用预览
        self.VK_RBUTTON = 0x02  # 右键

        # 获取屏幕的分辨率和中心点
        screen_width, screen_height = pyautogui.size()
        center_x, center_y = screen_width // 2, screen_height // 2
        self.screen_center = (center_x, center_y)  # 屏幕中心坐标
        self.width = 300
        self.left = center_x - self.width // 2
        self.top = center_y - self.width // 2
        self.height = self.width

        self.x_center, self.y_center = 0, 0
        self.sx, self.sy = 1, 1  # 鼠标移动灵敏度
        self.MOUSEEVENTF_MOVE = 0x0001

    def load_model(self, model_name):
        """ 加载模型 """
        if model_name != "暂无模型":
            print(f"加载模型: {model_name}")
            self.model = YOLO(model_name)
        else:
            print("未找到模型，无法加载")

    def toggle_preview(self):
        """ 切换预览状态 """
        self.preview_enabled = not self.preview_enabled
        print(f"预览已{'启用' if self.preview_enabled else '关闭'}")

    def is_right_mouse_button_pressed00(self):
        return ctypes.windll.user32.GetAsyncKeyState(self.VK_RBUTTON) & 0x8000 != 0

    def jietu(self):
        with mss.mss() as sct:
            region = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            threading.Thread(target=self.yolo(img), daemon=True).start()
            time.sleep(0.03)

    def yolo(self, img):
        print("检测开始...")
        if not self.model:
            print("模型未加载")
            return

        results = self.model(img)

        for result in results:
            boxes = result.boxes
            if boxes and len(boxes.xyxy) > 0:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()

                person_indices = np.where(class_ids == 0)[0]
                high_confidence_indices = person_indices[confidences[person_indices] > 0.50]

                if high_confidence_indices.size > 0:
                    highest_confidence_index = high_confidence_indices[np.argmax(confidences[high_confidence_indices])]
                    highest_confidence_box = boxes_data[highest_confidence_index]

                    # 计算目标框的中心点
                    x_center = int((highest_confidence_box[0] + highest_confidence_box[2]) / 2) + self.left
                    y_center = int((highest_confidence_box[1] + highest_confidence_box[3]) / 2) + self.top

                    self.x_center = x_center
                    self.y_center = y_center

                    # 画框标记目标
                    cv2.rectangle(img,
                                  (int(highest_confidence_box[0]), int(highest_confidence_box[1])),
                                  (int(highest_confidence_box[2]), int(highest_confidence_box[3])),
                                  (0, 255, 0), 2)  # 绿色框
                    cv2.putText(img,
                                f'Person: {confidences[highest_confidence_index]:.2f}',
                                (int(highest_confidence_box[0]), int(highest_confidence_box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # **计算目标框中心与屏幕中心的相对坐标**
                    relative_x = x_center - self.screen_center[0]
                    relative_y = y_center - self.screen_center[1]

                    # **移动鼠标到目标框相对坐标的位置**
                    if self.is_right_mouse_button_pressed00():
                        self.move_mouse_relative(relative_x, relative_y)

        # **无论是否检测到目标，都显示预览**
        if self.preview_enabled:
            print("显示实时预览...")
            try:
                cv2.imshow("Detection", img)
                cv2.waitKey(1)  # 维持窗口更新
                cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)  # 置顶窗口
            except cv2.error as e:
                print(f"预览窗口错误: {e}")
        else:
            cv2.destroyAllWindows()

    def move_mouse_relative(self, relative_x, relative_y):
        """ 根据目标框相对屏幕中心的坐标，移动鼠标 """
        screen_width, screen_height = pyautogui.size()

        # 使用灵敏度调整相对坐标
        move_x = int(relative_x * self.sx)
        move_y = int(relative_y * self.sy)

        # 计算移动步长
        steps = 5  # 步数
        step_x = move_x / steps
        step_y = move_y / steps

        # 用于累计移动量
        accumulated_x = 0.0
        accumulated_y = 0.0

        # print('相对坐标:', delta_x, delta_y)

        for i in range(steps):
            # 累加当前步长
            accumulated_x += step_x
            accumulated_y += step_y

            # 检查是否需要移动
            if abs(accumulated_x) >= 1:
                move_x = int(accumulated_x)
                ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_MOVE, move_x, 0, 0, 0)
                accumulated_x -= move_x  # 减去已移动的部分

            if abs(accumulated_y) >= 1:
                move_y = int(accumulated_y)
                ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_MOVE, 0, move_y, 0, 0)
                accumulated_y -= move_y  # 减去已移动的部分

            time.sleep(0.01)  # 小延迟以实现平滑运动

    def main(self):
        self.spot = False
        while not self.spot:
            self.jietu()
        else:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    root = ctk.CTk()
    app = AI_GUI(root)
    root.mainloop()
