import torch

"""
MPS available: True (can use Mac GPU), False (CPU only, or PyTorch not installed with MPS support).
"""
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())