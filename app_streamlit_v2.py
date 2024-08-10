import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import torch
import requests
from pptx import Presentation
from PIL import Image
import io
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.ndimage import zoom, median_filter as medfilt
import random
import pyautogui
import utils
def add_custom_css():
    st.markdown("""
        <style>
        /* Định dạng cho tiêu đề */
        h1 {
            font-size: 40px;
            color: rgb(3 41 106);
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Định dạng cho các nút bấm */
        button {
            background-color: rgb(3 41 106) !important;
            color: white !important;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #45a049 !important;
        }

        /* Định dạng cho các hộp kiểm (checkbox) */
        .stCheckbox label {
            font-size: 18px;
            color: #333;
        }

        /* Định dạng cho hình ảnh */
        img {
            border-radius: 15px;
            width: 100%;
            height: auto;
        }

        /* Định dạng cho cột trong Streamlit */
        .css-1lcbmhc {
            padding-top: 50px;
        }

        /* Định dạng nền tổng thể */
        .reportview-container {
            background-color: #f5f5f5;
            padding: 20px;
        }

        /* Định dạng footer */
        footer {
            text-align: center;
            padding: 10px;
            background-color: rgb(3 41 106);
            color: white;
            font-size: 14px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
def main():
    add_custom_css()
    image_path = 'logo.jpg'

    # Hiển thị hình ảnh trên Streamlit
    st.image(image_path, use_column_width=True)
    st.title("Webcam Live Feed with Hand Tracking")
    col1, col2, col3 = st.columns([1, 2, 1])
    st.title("Webcam Live Feed with Hand Tracking")

    # Tạo cột để chia bố cục trang
    col1, col2 = st.columns([1, 2])  # Cột trái (1 phần) và cột phải (2 phần)

    with col1:
        st.header("Video YouTube và Webcam")

        # Điều khiển YouTube
        video_url = st.text_input("Nhập URL video YouTube:")
        if video_url:
            st.video(video_url)
        else:
            st.write("Vui lòng nhập URL video YouTube.")

        # Điều khiển Webcam
        
    with col2:
        run = st.checkbox('Run')
        if video_url:
            st.video(video_url)
        else:
            st.write("Vui lòng nhập URL video YouTube.")
        FRAME_WINDOW = st.image([])
        C = utils.Config()
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        sequence = []
        predictions = []
        camera = cv2.VideoCapture(0)
    
        # Tạo một vùng trống để hiển thị nhãn
        prediction_placeholder = st.empty()
    with col3:
        while run:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoint = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    keypoint = np.array_split(keypoint, 22)
                    sequence.append(keypoint)
                    sequence = sequence[-60:]

                if len(sequence) == 60:
                    sequence_np = np.array(sequence, dtype=object)     
                    sequence_np = utils.pad_arrays(sequence_np)
                    X_test_rt_1, X_test_rt_2 = utils.data_generator_rt(sequence_np[-32:], C)

                    X_test_rt_1 = torch.from_numpy(X_test_rt_1).type(torch.FloatTensor)
                    X_test_rt_2 = torch.from_numpy(X_test_rt_2).type(torch.FloatTensor)
                    X_test_rt_1_list = X_test_rt_1.tolist()
                    X_test_rt_2_list = X_test_rt_2.tolist()

                    try:
                        response = requests.post("http://127.0.0.1:8000/predict", json={
                            "X_test_rt_1": X_test_rt_1_list,
                            "X_test_rt_2": X_test_rt_2_list
                        })
                        
                        if response.status_code == 200:
                            prediction = response.json().get("prediction", "Unknown")
                            predictions.append(prediction)
                            if len(predictions) > 1 and predictions[-1] != predictions[-2]:
                                # Cập nhật nhãn mới
                                prediction_placeholder.write(f"Prediction: {predictions[-1]}")
                                if predictions[-1] == "Shake":
                                    pyautogui.press('space')  # Tạm dừng video
                                if predictions[-1] == "SR":
                                    pyautogui.hotkey('right')
                                if predictions[-1] == "SL": 
                                    pyautogui.hotkey('left')  # Tua về s au 10 giây
                                if predictions[-1] == "SU":
                                    # Tăng âm lượng
                                    pyautogui.hotkey('up')  # Bạn cần kiểm tra phím tắt cụ thể cho hệ điều hành của bạn
                                elif predictions[-1] == "SD":
                                    # Giảm âm lượng
                                    pyautogui.hotkey('down')
                        else:
                            st.write("Error:", response.text)
                    except Exception as e:
                        st.write(f"Request failed: {str(e)}")
            FRAME_WINDOW.image(frame)
    

        camera.release()

if __name__ == "__main__":
    main()
