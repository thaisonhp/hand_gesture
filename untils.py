import cv2
import numpy as np
import torch
import requests
from scipy.ndimage import zoom
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
import json
from tqdm import tqdm
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Thay đổi kích thước, chuẩn hóa hoặc bất kỳ xử lý nào cần thiết
    # Ví dụ: chuẩn hóa pixel giá trị từ 0-255 sang 0-1
    image = image / 255.0
    # Đảm bảo hình dạng phù hợp với đầu vào mô hình
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    return image

def norm_train2d(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    return p

def data_generator_rt(T, C):
    X_0 = []
    X_1 = []
    print(T.shape)
    T = np.expand_dims(T, axis = 0)  # Thay đổi hình dạng dữ liệu để phù hợp với đầu vào của mô hình

    for i in tqdm(range(len(T))):
        p = np.copy(T[i])
        p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)

        M = get_CG(p, C)  # Tính toán các đặc trưng của dữ liệu
        X_0.append(M)
        p = norm_train2d(p)  # Chuẩn hóa dữ liệu
        X_1.append(p)

    X_0 = np.stack(X_0)  # Chuyển danh sách thành mảng numpy
    X_1 = np.stack(X_1)

    return X_0, X_1


def zoom_data(p, target_l, joints_num, joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape=[target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n], 3)
            p_new[:,m,n] = zoom(p[:,m,n], target_l/l)[:target_l]
    return p_new

def pad_arrays(data):
    max_length = max(max(arr.shape[0] for arr in frame) for frame in data)
    return np.array([[np.pad(arr, (0, max_length - arr.shape[0]), 'constant') for arr in frame] for frame in data])

def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M