from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGE(nn.Module):
    """



    """


    def __init__(
        self,
        e_num,
        r_num,
        relation_num,


        
    ):
        super().__init__()