# test_setup.py
import torch
import tiktoken
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
print("✓ Setup complete!")
