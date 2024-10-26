import threading
from PIL import Image
import keyboard
import mss
import ctypes
import pyautogui
import time
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller


class AI_Auto:
    def __init__(self):
        self.img = None  # 截图赋值
        self.model = YOLO('yolov8s.pt')  # 模型路径
        screen_width, screen_height = pyautogui.size()
        self.mouse = Controller()
        center_x = screen_width // 2
        center_y = screen_height // 2
        self.left = center_x - 160
        self.top = center_y - 160
        self.width = 320  # 截图大小
        self.height = self.width
        self.x_center = 0
        self.y_center = 0
        self.spot = False
        self.MOUSEEVENTF_MOVE = 0x0001
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2
        self.s = 0.50  # 灵敏度
        self.j_active = False  # 标志位，避免重复触发
        self.z_active = False  # 标志位，避免重复触发

        # 注册热键
        keyboard.add_hotkey('add', self.j)
        keyboard.add_hotkey('subtract', self.z)

    def jietu(self):
        with mss.mss() as sct:
            region = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}  # 屏幕中间截图区
            screenshot = sct.grab(region)
            img = Image.fromarray(np.array(screenshot))
            self.img = img

    def yolo(self):
        results = self.model(self.img)  # 识别图像
        for result in results:
            boxes = result.boxes

            if boxes is not None and len(boxes.xyxy) > 0:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()

                person_indices = np.where(class_ids == 0)[0]
                high_confidence_indices = person_indices[confidences[person_indices] > 0.8]  # 置信度0.8以上
                if high_confidence_indices.size > 0:
                    highest_confidence_index = high_confidence_indices[np.argmax(confidences[high_confidence_indices])]
                    highest_confidence_box = boxes_data[highest_confidence_index]

                    self.x_center = int((highest_confidence_box[0] + highest_confidence_box[2]) / 2) + self.left  # 计算识别框中心坐标
                    self.y_center = int((highest_confidence_box[1] + highest_confidence_box[3]) / 2) + self.top
                    return
                else:
                    print("没有检测到人类。")
                    self.x_center = 0
                    self.y_center = 0
            else:
                print("没有检测到任何对象。")
                self.x_center = 0
                self.y_center = 0

    def is_reft_mouse_button_pressed(self):
        return ctypes.windll.user32.GetAsyncKeyState(0x02) != 0  # 鼠标左键触发自瞄

    def move(self):
        while not self.spot:
            if self.is_reft_mouse_button_pressed():
                if self.x_center != 0 or self.y_center != 0:
                    # 打印目标坐标
                    print('目标坐标:', self.x_center, self.y_center)

                    # 计算移动坐标
                    delta_x = int((self.x_center - self.center_x) * self.s)
                    delta_y = int((self.y_center - self.center_y) * self.s)

                    # 计算距离
                    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5

                    # 动态设置步长
                    if distance > 100:  # 如果距离较远
                        move_x = int(delta_x * 0.1)  # 较大的步长
                        move_y = int(delta_y * 0.1)
                    elif distance > 20:  # 如果距离适中
                        move_x = int(delta_x * 0.05)  # 中等步长
                        move_y = int(delta_y * 0.05)
                    else:  # 如果距离很近
                        move_x = int(delta_x * 0.02)  # 小步长
                        move_y = int(delta_y * 0.02)

                    # 使用 mouse_event 移动鼠标
                    ctypes.windll.user32.mouse_event(self.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

                    time.sleep(0.01)  # 控制移动速度

    def end(self):
        keyboard.add_hotkey('f5', self.on_f5_press)  #  F5退出

    def on_f5_press(self):
        self.spot = True

    def j(self):
        if not self.j_active:  # 数字小键盘加减灵敏度
            self.j_active = True
            self.s += 0.01
            print('灵敏度', self.s)
            time.sleep(0.5)
            self.j_active = False

    def z(self):
        if not self.z_active:  # 数字小键盘加减灵敏度
            self.z_active = True
            self.s = max(0.01, self.s - 0.01)
            print('灵敏度', self.s)
            time.sleep(0.5)
            self.z_active = False

    def run(self):
        while True:
            if self.spot:
                print("F5 程序终止")
                break
            self.jietu()
            self.yolo()
            time.sleep(0.04)

    def main(self):
        t_end = threading.Thread(target=self.end)
        t_end.start()
        t_move = threading.Thread(target=self.move)
        t_move.start()
        self.run()


if __name__ == '__main__':
    y = AI_Auto()
    y.main()
