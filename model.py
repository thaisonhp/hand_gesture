import torch
import models.DDNet_Original as Net

def load_model():
    # Tạo lại cấu trúc mô hình
    model = Net.DDNet_Original(frame_l=32, joint_n=22, joint_d=3, class_num=14, feat_d=231, filters=64)
    device = torch.device('cpu')  # Chỉ định sử dụng CPUe.
    # Tải trạng thái mô hình
    model.load_state_dict(torch.load('model.pt', map_location=device))
    # Đặt mô hình ở chế độ đánh giá
    model.eval()
    
    return model
