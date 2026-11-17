from ultralytics import YOLO
from glob import glob
import os

root_dir = "/home/gpuadmin/repo/yolov11/ultralytics/runs/classify"
dataset = "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/PersonDrownClsDataset_v0.1.0"
target_dir = glob(f"{root_dir}/train*")

fitness = 0
best_model = None  # 초기화

for target in target_dir:
    file_name = os.path.join(target, "weights/best.pt")
    if os.path.isfile(file_name):
        try:
            model = YOLO(file_name)
            
            val_result = model.val(data=dataset, split="val")
            val_now_fitness = val_result.fitness
            
            test_result = model.val(data=dataset, split="test")
            test_now_fitness = test_result.fitness
            
            balance = 0.2 * val_now_fitness + 0.8 * test_now_fitness
            
            if balance > fitness:
                fitness = balance
                best_model = file_name
                print(f"now Model: {file_name}, fitness: {balance:.4f}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if best_model:
    print(f"✅ Best model: {best_model}, fitness: {fitness:.4f}")
else:
    print("❌ No valid model found.")
