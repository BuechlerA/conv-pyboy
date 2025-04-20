import torch
import logging
import os

# Logger for device utilities
logger = logging.getLogger(__name__)

def get_device():
    """
    Get the appropriate device for PyTorch operations.
    Returns MPS device if available (Apple Silicon),
    CUDA if available (NVIDIA GPU),
    otherwise falls back to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # silent: device selection logged once in print_device_info
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # silent: will be logged in print_device_info with PID
    else:
        device = torch.device("cpu")
        # silent: will be logged in print_device_info with PID
    
    return device

def print_device_info():
    """
    Print information about the PyTorch device being used
    """
    device = get_device()
    pid = os.getpid()
    
    if device.type == "mps":
        logger.info(f"[PID {pid}] Apple Silicon (MPS) for computation")
    elif device.type == "cuda":
        logger.info(f"[PID {pid}] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"[PID {pid}] CPU for computation")