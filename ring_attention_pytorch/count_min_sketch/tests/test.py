import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Name of GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available to PyTorch after the recent installation.")
    print("This could be due to a few reasons even after a successful pip install:")
    print("  1. NVIDIA drivers might still have issues or are not compatible with the CUDA version PyTorch expects.")
    print("     - You can check drivers with `nvidia-smi` in your terminal.")
    print("  2. The system-wide CUDA toolkit might be missing, or the one PyTorch found is not compatible.")
    print("  3. The CUDA version in the pip URL (cu128) might not perfectly match your system's capabilities or a standard PyTorch distribution. PyTorch usually has wheels for versions like cu118 (CUDA 11.8) or cu121 (CUDA 12.1). 'cu128' is unusual.")
    print("     - Double-check the exact CUDA version your NVIDIA driver supports (`nvidia-smi` can give clues).")
    print("     - Visit https://pytorch.org/get-started/locally/ to get the precise pip command for your system's CUDA version.")
