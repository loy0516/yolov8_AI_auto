import threading
import cv2
import keyboard
import mss
import ctypes
import pyautogui
import time
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller
import queue


class AI_Auto:
    def __init__(self):
        # self.img = None
        self.model = YOLO('YOLOv8s_apex_teammate_enemy.pt')
        screen_width, screen_height = pyautogui.size()
        self.mouse = Controller()
        center_x = screen_width // 2
        center_y = screen_height // 2
        self.left = center_x - 160
        self.top = center_y - 160
        self.width = 320
        self.height = self.width
        self.x_center = 0
        self.y_center = 0
        self.x_center_b = 0
        self.y_center_b = 0
        self.spot = False
        self.MOUSEEVENTF_MOVE = 0x0001
        self.sx = 0.50
        self.sy = 0.25
        self.j_active = False  # 标志位，避免重复触发
        self.z_active = False  # 标志位，避免重复触发

        # 注册热键
        keyboard.add_hotkey('add', self.j)
        keyboard.add_hotkey('subtract', self.z)

    def jietu(self):
        with mss.mss() as sct:
            region = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            screenshot = sct.grab(region)
            img = np.array(screenshot)

            # 转换为 BGR 格式（OpenCV 使用的格式）
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.yolo(img)

    def yolo(self, img__):
        results = self.model(img__)

        for result in results:
            boxes = result.boxes

            if boxes is not None and len(boxes.xyxy) > 0:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()

                person_indices = np.where(class_ids == 0)[0]
                high_confidence_indices = person_indices[confidences[person_indices] > 0.75]

                if high_confidence_indices.size > 0:
                    highest_confidence_index = high_confidence_indices[np.argmax(confidences[high_confidence_indices])]
                    highest_confidence_box = boxes_data[highest_confidence_index]

                    # 计算中心点
                    self.x_center = int((highest_confidence_box[0] + highest_confidence_box[2]) / 2) + self.left
                    self.y_center = int((highest_confidence_box[1] + highest_confidence_box[3]) / 2) + self.top

                    # 绘制识别框
                    cv2.rectangle(img__,
                                  (int(highest_confidence_box[0]), int(highest_confidence_box[1])),
                                  (int(highest_confidence_box[2]), int(highest_confidence_box[3])),
                                  (255, 0, 0), 2)  # 红色框
                    cv2.putText(img__,
                                f'Person: {confidences[highest_confidence_index]:.2f}',
                                (int(highest_confidence_box[0]), int(highest_confidence_box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    break  # 找到一个人后，可以选择退出循环
                else:
                    print("未检测到高置信度的人类。")
                    # self.x_center = 0
                    # self.y_center = 0
            else:
                print("未检测到任何对象。")
                # self.x_center = 0
                # self.y_center = 0

        # 实时显示图像
        cv2.imshow('Detection', img__)
        cv2.waitKey(1)  # 等待1毫秒以更新图像

    def is_reft_mouse_button_pressed(self):
        return ctypes.windll.user32.GetAsyncKeyState(0x02) != 0

    def move(self):
        while not self.spot:
            if self.is_reft_mouse_button_pressed():
                if self.x_center != self.x_center_b or self.y_center != self.y_center_b:
                    if self.x_center != 0 or self.y_center != 0:
                        self.x_center_b = self.x_center
                        self.y_center_b = self.y_center
                        screen_width, screen_height = pyautogui.size()
                        center_x = screen_width // 2
                        center_y = screen_height // 2
                        # 打印目标坐标
                        print('目标坐标:', self.x_center, self.y_center)

                        # 计算目标的相对坐标
                        delta_x = (self.x_center - center_x) * self.sx
                        delta_y = (self.y_center - center_y) * self.sy

                        # 计算移动步长
                        steps = 20  # 步数
                        step_x = delta_x / steps
                        step_y = delta_y / steps

                        # 用于累计移动量
                        accumulated_x = 0.0
                        accumulated_y = 0.0

                        print('相对坐标:', delta_x, delta_y)

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

    def end(self):
        keyboard.add_hotkey('f5', self.on_f5_press)

    def on_f5_press(self):
        self.spot = True

    def j(self):
        if not self.j_active:
            self.j_active = True
            self.sx += 0.01
            print('灵敏度', self.sx)
            time.sleep(0.5)
            self.j_active = False

    def z(self):
        if not self.z_active:
            self.z_active = True
            self.sx = max(0.01, self.sx - 0.01)
            print('灵敏度', self.sx)
            time.sleep(0.5)
            self.z_active = False

    def run(self):
        while True:
            if self.spot:
                print("F5 程序终止")
                break
            self.jietu()
            # self.yolo()
            # time.sleep(0.04)

    def main(self):
        t_end = threading.Thread(target=self.end)
        t_end.start()
        t_move = threading.Thread(target=self.move)
        t_move.start()
        self.run()


if __name__ == '__main__':
    y = AI_Auto()
    y.main()
