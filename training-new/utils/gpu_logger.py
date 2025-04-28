import torch
import pynvml
import psutil

def log_gpu(threshold_warning_gb=4.0):
    print("-" * 30)  # Separator for clarity
    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            print(f"Checking memory for GPU: {device_name} (Index: {device_index})")

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem_gb = meminfo.total / (1024 ** 3)
            free_mem_gb = meminfo.free / (1024 ** 3)

            allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
            peak_allocated = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)

            print(f"Total VRAM: {total_mem_gb:.2f} GiB")
            print(f"Free VRAM: {free_mem_gb:.2f} GiB")
            print(f"Memory allocated by PyTorch: {allocated:.2f} GiB")
            print(f"Memory reserved by PyTorch:  {reserved:.2f} GiB")
            print(f"Peak memory used (PyTorch):  {peak_allocated:.2f} GiB")

            if free_mem_gb < threshold_warning_gb:
                print(f"⚠️  WARNING: Low available VRAM ({free_mem_gb:.2f} GiB) — evaluation might crash.")

        except Exception as e:
            print(f"Could not get GPU memory info. Error: {e}")

    else:
        print("CUDA not available, cannot check GPU memory.")

    # System RAM check
    mem = psutil.virtual_memory()
    print(f"System RAM usage: {mem.percent:.2f}% ({mem.used / (1024 ** 3):.2f} GiB / {mem.total / (1024 ** 3):.2f} GiB)")
    print("-" * 30)  # End separator
