from ultralytics import YOLO


# model = YOLO("/home/gpuadmin/repo/KISA/kisa-od-model-trainer/hubs/PersonDrownCls_v0.1.0/weights/best.pt")
# result = model.val(data="/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/PersonDrownKurClsDataset_v0.1.0", split = "test")

model = YOLO("/home/gpuadmin/repo/yolov11/ultralytics/hubs/PersonWalljumpCls_v2.3.0/artifacts/weights/best_no_transform.pt")
result = model.val(data="/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/WallJumpClsDataset_v0.2.0", split = "val")
print(result)
result = model.val(data="/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/WallJumpClsDataset_v0.2.0", split = "test")
print(result)