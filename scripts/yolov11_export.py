from ultralytics import YOLO
import cv2
import time

# model_name = "yolo11s"
model_name = "/home/gpuadmin/repo/yolov11/ultralytics/hubs/PersonDet_v3.0.0/artifacts/weights/best"
pytorch_model = YOLO(f"{model_name}.pt").cuda()
pytorch_model.export(format="engine", half=True, dynamic=True, batch=4, device=0)
 
onnx_model = YOLO(f"{model_name}.onnx")
engine_model = YOLO(f"{model_name}.engine")
 
cap = cv2.VideoCapture("/home/gpuadmin/Downloads/simple.mp4")
torch_spend_time_avg = 0.
engine_spend_time_avg = 0.
onnx_spend_time_avg = 0.
 
cnt = 0
while True :
    ret, frame = cap.read()
    if ret :
        cnt += 1
        t1 = time.time()
        pytorch_model.predict(frame, verbose=False)
        t2 = time.time()
        engine_model.predict(frame, verbose=False)
        t3 = time.time()
        onnx_model.predict(frame, verbose=False)
        onnx_spend_time = time.time() - t3
        torch_spend_time = t2-t1
        engine_spend_time = t3-t2
 
        torch_spend_time_avg += torch_spend_time
        onnx_spend_time_avg += onnx_spend_time
        engine_spend_time_avg += engine_spend_time
 
        print(f"engine : {1000 * engine_spend_time:.2f}ms\ttorch : {1000 * torch_spend_time:.2f}ms\tonnx : {1000 *onnx_spend_time:.2f}ms")
 
    else : break
 
print(f"""
모델 : {model_name}
전체 걸린 시간의 평균 계산
전체 프레임 개수 : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}
분석한 프레임 개수 : {cnt}
pytorch inference에 걸린 평균 시간 : {torch_spend_time_avg / cnt * 1000:.2f}ms
ONNX inference에 걸린 평균 시간 : {onnx_spend_time_avg / cnt * 1000:.2f}ms
TensorRT(engine) inference에 걸린 평균 시간 : {engine_spend_time_avg / cnt * 1000:.2f}ms
""")