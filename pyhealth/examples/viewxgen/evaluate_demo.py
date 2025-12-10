import torch
from model_demo import MiniViewXGen

model = MiniViewXGen()

tokens = torch.tensor([[12, 57, 91, 33]])

output = model(tokens)

print("Output shape:", output.shape)
print("Evaluation script executed successfully.")
