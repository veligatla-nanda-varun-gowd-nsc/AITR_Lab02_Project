"""
verify_env.py
-------------
Audits hardware compute resources, GPU VRAM allocation, and verifies
4-bit quantization kernel compatibility prior to running inference.
"""

import sys
import torch

def inspect_hardware():
    print("=" * 60)
    print("      RESEARCH ENVIRONMENT & COMPUTE AUDIT")
    print("=" * 60)
    
    # 1. Python & PyTorch Core Check
    print(f"[+] Python Version  : {sys.version.split()[0]}")
    print(f"[+] PyTorch Version : {torch.__version__}")
    
    # 2. CUDA Device Inspection
    cuda_ready = torch.cuda.is_available()
    print(f"[+] CUDA Available  : {cuda_ready}")
    
    if not cuda_ready:
        print("[!] WARNING: No CUDA GPU detected. Running in 4-bit requires an NVIDIA GPU.")
        print("[!] Inference on CPU will fallback to float32 and will be very slow.")
        return False
        
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    
    print(f"[+] GPU Identifier  : {gpu_name}")
    print(f"[+] Total GPU VRAM  : {total_memory:.2f} GB")
    
    # Enforce minimum memory requirement for 7B 4-bit (~5.5 GB)
    if total_memory < 6.0:
        print("[!] CAUTION: Detected VRAM is under 6.0 GB. You may experience CUDA OOM.")
    else:
        print("[+] Hardware Check  : PASS (VRAM sufficient for 4-bit Qwen2.5-7B).")

    # 3. BitsAndBytes 4-Bit Kernel Sanity Check
    print("\n[+] Testing 4-bit quantization kernel support...")
    try:
        import bitsandbytes as bnb
        linear_test = bnb.nn.Linear4bit(
            128, 128, 
            bias=False, 
            compute_dtype=torch.bfloat16, 
            quant_type="nf4"
        ).cuda()
        dummy_input = torch.randn(2, 128, dtype=torch.bfloat16).cuda()
        _ = linear_test(dummy_input)
        print("[+] BitsAndBytes    : PASS (4-bit NF4 kernels functioning properly).")
    except Exception as e:
        print(f"[!] FAILED: bitsandbytes 4-bit test threw an exception: {e}")
        return False

    # 4. Hugging Face Datasets & Transformers Import Test
    print("\n[+] Testing Hugging Face core imports...")
    try:
        import transformers
        import datasets
        print(f"[+] Transformers    : v{transformers.__version__} (OK)")
        print(f"[+] Datasets        : v{datasets.__version__} (OK)")
    except ImportError as e:
        print(f"[!] Import Error    : {e}")
        return False

    print("=" * 60)
    print("Status: Environment is fully verified and ready for Prompt 09.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = inspect_hardware()
    if not success:
        sys.exit(1)
