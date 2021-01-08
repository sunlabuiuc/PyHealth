""" Miscellaneous utils. """
import random
import logging
import torch
import numpy as np


def set_seed(seed_value):
    """ Sets the random seed to a given value. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info("Random seed: %d", seed_value)
