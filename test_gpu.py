#!/usr/bin/env python3
import torch
import sys
import subprocess

print("="*60)
print("GPU Test Script")
print("="*60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try to run nvidia-smi
print("\n" + "-"*60)
print("nvidia-smi output:")
print("-"*60)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"Could not run nvidia-smi: {e}")

print("-"*60)

if torch.cuda.is_available():
    print(f"\n✓ CUDA version: {torch.version.cuda}")
    print(f"✓ CUDA device count: {torch.cuda.device_count()}")
    print(f"✓ CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
    
    # Test actual GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"✓ GPU matrix multiplication successful!")
    print(f"✓ Result shape: {z.shape}")
    print(f"✓ Result device: {z.device}")
    
    print("\n" + "="*60)
    print("SUCCESS! GPU is working correctly!")
    print("="*60)
else:
    print("\n✗ WARNING: CUDA is not available!")
    print("This script should only run on GPU nodes.")
    print("="*60)
