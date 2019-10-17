import torch


def best_available_device():
    if torch.cuda.device_count():
        return 'cuda'
    return 'cpu'
