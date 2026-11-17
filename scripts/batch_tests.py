import torch
import time
from ultralytics.nn.autobackend import AutoBackend

MODEL_PATH = "/home/gpuadmin/repo/yolov11/ultralytics/hubs/PersonGenderCls_v3.2.0/PersonGenderCls_v3.2.0.pt"

def print_gpu_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{prefix}Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def load_model(yolo_path):
    print_gpu_mem("Before model load:")
    model = AutoBackend(weights=yolo_path, device=torch.device("cuda:1"), verbose=False)
    print_gpu_mem("After model load:")
    return model

def make_torch_224_rand_batch(batch_size=8):
    return torch.randn(batch_size, 3, 224, 224).to("cuda:1")

def test_batch_memory(model, num_batches=5, batch_size=2):
    for i in range(num_batches):
        time.sleep(1)
        print(f"\nBatch {i+1}:")
        print_gpu_mem("Before inference: ")
        x = make_torch_224_rand_batch(batch_size)
        with torch.no_grad():
            _ = model(x)
        print_gpu_mem("After inference: ")
    del model
    del x
    torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = load_model(MODEL_PATH)
    test_batch_memory(model, num_batches=10, batch_size=2048)
