import numpy as np
import os
import random
import torch
from typing import List, Optional


def init_random_seed_torch(value=0) -> None:
    """Initializes random seed for reproducibility random processes
    Args:
        value:

    Returns:
        None
    """
    random.seed(value)
    np.random.seed(value)
    os.environ['PYTHONHASHSEED']=str(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


def gpus_to_use(gpus_list: Optional[List] = None) -> None:
    """Chooses GPUs to use
    Args:
        gpus_list: GPUs index list

    Returns:
        None
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus_list is not None:
        gpus_list = [str(item) for item in gpus_list]
        gpus = ",".join(gpus_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpus}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_device(devices_list: Optional[List] = None) -> torch.device:
    """
    Args:
        devices_list: GPUs index list

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        gpus_to_use(devices_list)
        return torch.device("cuda")
    else:
        return torch.device("cpu")
