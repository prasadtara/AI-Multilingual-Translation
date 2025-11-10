# test_setup.py
import torch
import transformers
from transformers import pipeline

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

print("\nTransformers version:", transformers.__version__)
print("\nSetup successful!")

# Quick test of translation pipeline (uses CPU, just to verify)
print("\nTesting translation pipeline...")
translator = pipeline("translation_en_to_fr", model="t5-small")
result = translator("Hello, how are you?")
print("Test translation:", result)
