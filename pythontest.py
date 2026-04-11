import torch
import torch.nn as nn

x = torch.randn(4, 10, 8)  # batch, time, features

lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
out, _ = lstm(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)