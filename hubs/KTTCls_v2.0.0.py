from ultralytics import YOLO
import time
import tqdm
import ultralytics.data.augment as augment
augment.CenterCrop = augment.ClassifyLetterBox

transforms = [
    augment.ClassifyLetterBox(size=224),
    augment.ToTensor(),
]

try:
    import torch
    torch.cuda.empty_cache()
    t0 = time.time()
    model = YOLO("yolo11l-cls.pt", task="classify")
    t1 = time.time()
    model.train(data='cifar10', transforms=transforms)
    t2 = time.time()
    print(f"model_load: {(t1-t0):.2f}s, train: {(t2-t1):.2f}s")
    
except Exception as e:
    print(e)
    raise e
