from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
from model import load_model
import logging

# Cấu hình logging để theo dõi lỗi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ định các nguồn gốc được phép
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Config():
    def __init__(self):
        self.frame_l = 32  # chiều dài của khung hình
        self.joint_n = 22  # số lượng khớp
        self.joint_d = 3   # kích thước của khớp
        self.clc_num = 14  # số lượng lớp
        self.feat_d = 231
        self.filters = 64

class PredictionRequest(BaseModel):
    X_test_rt_1: list
    X_test_rt_2: list

C = Config()

model = load_model()
if model is None:
    logger.error("Model loading failed.")
    raise RuntimeError("Model loading failed.")

@app.get("/")
def hello():
    return "Hello world"

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        sentences = []
        labels = ['Tap', 'Grab', 'RC', 'Pinch', 'Expand', 'RCC', 'SR', 'SL', 'SD', 'SU', 'SV', 'S+', 'SX', 'Shake']
        # Chuyển đổi danh sách thành numpy arrays và sau đó thành PyTorch tensors
        
        X_test_rt_1 = torch.tensor(np.array(request.X_test_rt_1)).float()
        X_test_rt_2 = torch.tensor(np.array(request.X_test_rt_2)).float()
        # Print shapes for debugging

        print("X_test_rt_1 shape:", X_test_rt_1.shape)
        print("X_test_rt_2 shape:", X_test_rt_2.shape)

        with torch.no_grad():
            Y_pred = model(X_test_rt_1, X_test_rt_2).cpu().numpy()
        sentences.append(labels[np.argmax(Y_pred)])
        return {"prediction":sentences[-1]}
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
