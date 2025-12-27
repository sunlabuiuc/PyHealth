# pyhealth/datasets/labtop.py
import torch
from torch.utils.data import Dataset

class LabTOPDataset(Dataset):
    """
    Dataset for LabTOP sequences.
    Converts sequences of lab events (and demographics) into fixed-length tensors.
    """
    def __init__(self, sequences, tokenizer, max_len=128):
        self.samples = []
        self.max_len = max_len
        for seq in sequences:
            flat_seq = []
            for e in seq:
                flat_seq.extend(tokenizer.encode_event(**e))
            if len(flat_seq) < 2:
                continue
            inp = flat_seq[:-1]
            tgt = flat_seq[1:]

            # Fixed-length padding / truncation
            inp = (inp + [0]*self.max_len)[:self.max_len]
            tgt = (tgt + [0]*self.max_len)[:self.max_len]

            self.samples.append((torch.tensor(inp), torch.tensor(tgt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
