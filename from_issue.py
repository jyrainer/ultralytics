from ultralytics import YOLO
import ultralytics.data.augment as augment

# Create custom transforms
transforms = [
    augment.ClassifyLetterBox(size=224),
    augment.ToTensor(),
]

model = YOLO('/home/gpuadmin/repo/yolov11/ultralytics/ultralytics/cfg/models/11/yolo11-cls.yaml')
model.train(data='cifar10', transforms=transforms)