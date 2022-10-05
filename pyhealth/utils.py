import logging
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

try:
    import pynvml  # provides utility for NVIDIA management

    HAS_NVML = True
except:
    HAS_NVML = False


def collate_fn_dict(batch):
    """collate function for dict data"""
    return {key: [d[key] for d in batch] for key in batch[0]}


def set_logger(output_path: Optional[str] = None, exp_name: Optional[str] = None):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # streamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    # fileHandler
    if output_path is None:
        output_path = "./"
    if exp_name is None:
        exp_name = datetime.now().strftime("%y%m%d-%H%M%S")
    exp_path = os.path.join(output_path, exp_name)
    create_directory(exp_path)
    log_filename = os.path.join(exp_path, "log.txt")
    file_handler = logging.FileHandler(log_filename)
    logger.addHandler(file_handler)
    return exp_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def get_device(enable_cuda):
    cuda = torch.cuda.is_available() & enable_cuda
    if cuda:
        divice_index = auto_select_device()
        device = torch.device(f"cuda:{divice_index}")
    else:
        device = torch.device("cpu")
    logging.debug(f"Device: {device}")
    return device


def auto_select_device():
    """select gpu which has the largest free memory"""
    if HAS_NVML:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        logging.debug(f"Found {deviceCount} GPUs")
        largest_free_mem = 0
        largest_free_idx = 0
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = info.free / 1024.0 / 1024.0  # convert to MB
            total_mem = info.total / 1024.0 / 1024.0
            logging.debug(f"GPU {i} memory: {free_mem:.0f}MB / {total_mem:.0f}MB")
            if free_mem > largest_free_mem:
                largest_free_mem = free_mem
                largest_free_idx = i
        pynvml.nvmlShutdown()
        logging.debug(
            f"Using largest free memory GPU {largest_free_idx} with free memory {largest_free_mem:.0f}MB"
        )
        return str(largest_free_idx)
    else:
        logging.warning("pynvml is not installed, device auto-selection is disabled!")
        return "0"
