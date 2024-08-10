import cv2
import mediapipe as mp
import pyautogui


# Hàm để điều khiển YouTube
def control_youtube(label):
    if label != "":
        pyautogui.press('space')  # Phím tắt để phát/tạm dừng video
    else:
        print("Label không hợp lệ!")
