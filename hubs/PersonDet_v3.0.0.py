from ultralytics import YOLO
import torch
import time
import os
HUB_DIR ="/home/gpuadmin/repo/yolov11/ultralytics/hubs"
try:
    torch.cuda.empty_cache()
    cfg_path = "PersonDet_v3.0.0.yaml"
    
    cfg = os.path.join(HUB_DIR, cfg_path)
    t0 = time.time()
    model = YOLO("yolo11s.pt", task="detect")
    t1 = time.time()
    model.train(cfg=cfg)
    t2 = time.time()
    del model

    print(f"model_load: {(t1-t0):.2f}s, train: {(t2-t1):.2f}m")

except Exception as e:
    print(e)
    raise e

