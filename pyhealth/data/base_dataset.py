import random
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod
import torch


class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
