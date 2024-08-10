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

def control_youtube(label):
    if label == "SR":
        pyautogui.hotkey('shift', 'left')  # Tua về trước 10 giây (thay đổi phím tắt nếu cần)
    elif label == "SL":
        pyautogui.hotkey('shift', 'right')  # Tua về sau 10 giây (thay đổi phím tắt nếu cần)
    elif label == "Shake":
        pyautogui.press('space')  # Tạm dừng video
    elif label == "SU":
        pyautogui.hotkey('volumeup')  # Tăng âm lượng (thay đổi nếu cần)
    elif label == "SD":
        pyautogui.hotkey('volumedown')  # Giảm âm lượng (thay đổi nếu cần)
    else:
        print("Label không hợp lệ hoặc không có hành động tương ứng!")
class Config:
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_num = 14  # the number of class
        self.feat_d = 231
        self.filters = 64

def pad_arrays(data):
    max_length = max(max(arr.shape[0] for arr in frame) for frame in data)
    return np.array([[np.pad(arr, (0, max_length - arr.shape[0]), 'constant') for arr in frame] for frame in data])

def zoom_and_filter(p, target_l, joints_num, joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape=[target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new

def sampling_frame(p, C):
    full_l = p.shape[0]
    if random.uniform(0, 1) < 0.5:  # alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l
        p = p[int(s):int(e), :, :]
    else:  # without alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom_and_filter(p, C.frame_l, C.joint_n, C.joint_d)
    return p

def norm_train(p):
    p[:, :, 0] -= np.mean(p[:, :, 0])
    p[:, :, 1] -= np.mean(p[:, :, 1])
    p[:, :, 2] -= np.mean(p[:, :, 2])
    return p

def norm_train2d(p):
    p[:, :, 0] -= np.mean(p[:, :, 0])
    p[:, :, 1] -= np.mean(p[:, :, 1])
    return p

def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M

def data_generator_rt(T, C):
    X_0 = []
    X_1 = []
    T = np.expand_dims(T, axis=0)
    for i in tqdm(range(len(T))):
        p = np.copy(T[i])
        p = zoom_and_filter(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
        M = get_CG(p, C)
        X_0.append(M)
        p = norm_train2d(p)
        X_1.append(p)
    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    return X_0, X_1
