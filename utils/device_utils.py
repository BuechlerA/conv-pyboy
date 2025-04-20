import torch

def get_device():
    """
    Get the appropriate device for PyTorch operations.
    Returns MPS device if available (Apple Silicon),
    CUDA if available (NVIDIA GPU),
    otherwise falls back to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device

def print_device_info():
    """
    Print information about the PyTorch device being used
    """
    device = get_device()
    
    if device.type == "mps":
        print(f"Using Apple Silicon (MPS) for computation")
    elif device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for computation")