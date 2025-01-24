from ultralytics import YOLO
import time
import tqdm

try:
    import torch
    torch.cuda.empty_cache()
    t0 = time.time()
    model = YOLO("yolo11m-cls.pt", task="classify")
    t1 = time.time()
    model.train(cfg="/home/gpuadmin/repo/yolov11/ultralytics/hubs/KTTCls_v2.0.0.yaml")
    t2 = time.time()
    print(f"model_load: {(t1-t0):.2f}s, train: {(t2-t1):.2f}s")
    
except Exception as e:
    print(e)
    raise e
