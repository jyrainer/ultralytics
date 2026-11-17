from ultralytics import YOLO

dataset_dict = {
    "yonsei": "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiWallDataset_v0.1.0/exports/ULTRALYTICS/data.yaml",
    "aihub": "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/AihubWallDataset_v0.1.0/exports/ULTRALYTICS/data.yaml",
    "person": "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/PersonDataset_v0.2.0/exports/ULTRALYTICS/data.yaml"
}

model_dict = {
    "205": "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/hubs/PersonDet_v2.0.5/weights/best_ckpt.pt",
    "212": "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/hubs/PersonDet_v2.1.2/weights/best_ckpt.pt",
    "312": "/home/gpuadmin/repo/yolov11/ultralytics/hubs/PersonDet_v3.1.2/artifacts/weights/best.pt",
    "313": "/home/gpuadmin/repo/yolov11/ultralytics/hubs/PersonDet_v3.1.3/artifacts/weights/PersonDet_v3.1.3.pt"
}

model = YOLO(model_dict["205"])
# result = model.val(data=dataset_dict["person"],
#                    batch = 128, conf = 0.35, iou = 0.5, imgsz=640,split = "train")
# print(result.box.map)
result = model.val(data=dataset_dict["aihub"],
                   batch = 128, conf = 0.35, iou = 0.5, imgsz=640, split = "train")
print(result.box.map)
result = model.val(data=dataset_dict["yonsei"],
                   batch = 128, conf = 0.35, iou = 0.5, imgsz=640, split = "train")
print(result.box.map)

del model

model = YOLO(model_dict["212"])
result = model.val(data=dataset_dict["person"],
                   batch = 128, conf = 0.35, iou = 0.5, imgsz=640,split = "train")
print(result.box.map)
result = model.val(data=dataset_dict["aihub"],
                   batch = 128, conf = 0.35, iou = 0.5, imgsz=640, split = "train")
print(result.box.map)
result = model.val(data=dataset_dict["yonsei"],
                   batch = 128, conf = 0.35, iou = 0.5, imgsz=640, split = "train")
print(result.box.map)