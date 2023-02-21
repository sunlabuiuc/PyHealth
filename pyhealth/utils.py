import os
import pickle
import random

import numpy as np
import torch

from pyhealth import BASE_CACHE_PATH


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

# used for aggregating pickles created.
def record_dataset_cache(repo_root, filepath_to_record):
    f = open(os.path.join(repo_root, "pickled_datasets.txt"), "a")
    f.write(filepath_to_record)
    f.write("\n")
    f.close()