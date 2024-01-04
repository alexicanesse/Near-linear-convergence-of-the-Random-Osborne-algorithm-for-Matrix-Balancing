import numpy as np

try:
    import cupy as cp
    has_cuda = True
except ImportError:
    has_cuda = False


