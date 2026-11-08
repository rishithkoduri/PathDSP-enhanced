"""
Centralized Utilities for PathDSP-PIU:
- Reproducibility & deterministic seed control
- Hardware and GPU memory tracking
"""

import random
import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Enforces deterministic random seed across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    logger.debug(f"Deterministic seed set to {seed} (deterministic={deterministic})")


def get_gpu_memory_info(device: torch.device = None):
    """
    Returns GPU memory usage in MB if CUDA is available.
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    dev = device if device is not None and device.type == "cuda" else torch.device("cuda:0")
    allocated = torch.cuda.memory_allocated(dev) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    return {
        "device": torch.cuda.get_device_name(dev),
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "max_allocated_mb": round(max_allocated, 2)
    }
