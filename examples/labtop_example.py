# pyhealth/examples/labtop_example.py
import torch
from torch.utils.data import DataLoader
from pyhealth.datasets.labtop import LabTOPDataset
from pyhealth.models.labtop_transformer import LabTOPTransformer
from pyhealth.tasks.labtop_next_token import LabTOPNextTokenTask

# -----------------------------
# Minimal Dummy Tokenizer
# -----------------------------
class DummyTokenizer:
    """A minimal tokenizer for demonstration purposes."""
    def __init__(self):
        self.vocab_size = 100

    def encode_event(self, **event):
        return [1,2,3,4,5]  # fixed dummy sequence

# -----------------------------
# Dummy sequences
# -----------------------------
def build_dummy_sequences():
    return [
        [{"lab":"AGE3","value":"0","unit":"unit0","day":0,"weekday":0,"hour":0,"minute":0},
         {"lab":"50868","value":"1.20","unit":"unit1","day":0,"weekday":1,"hour":10,"minute":30}],
        [{"lab":"AGE5","value":"0","unit":"unit0","day":0,"weekday":0,"hour":0,"minute":0},
         {"lab":"50912","value":"0.80","unit":"unit1","day":0,"weekday":2,"hour":9,"minute":15}]
    ]

# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# -----------------------------
# Main
# -----------------------------
def main():
    tokenizer = DummyTokenizer()
    sequences = build_dummy_sequences()
    dataset = LabTOPDataset(sequences, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    model = LabTOPTransformer(vocab=tokenizer.vocab_size, d_model=64, nhead=2, nlayers=2, max_len=32)
    task = LabTOPNextTokenTask()

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    logits = model(input_ids)
    loss = task.get_loss(logits, labels)

    print("Logits shape:", logits.shape)
    print("Loss value:", loss.item())
    print("LabTOP example ran successfully.")

if __name__ == "__main__":
    main()
