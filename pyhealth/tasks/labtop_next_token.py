# pyhealth/tasks/labtop_next_token_task.py
import torch
import torch.nn as nn

class LabTOPNextTokenTask:
    """
    Task wrapper for LabTOP next-token prediction.
    Computes loss for sequence modeling.
    """
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def get_loss(self, logits, labels):
        B, L, V = logits.shape
        loss = self.loss_fn(logits.view(B*L, V), labels.view(B*L))
        return loss
