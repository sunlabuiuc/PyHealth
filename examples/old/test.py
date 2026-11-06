import torch

for i in range(100):
    a = torch.randn((5, 5, 5)) > 0
    print(((torch.sum(a, dim=2) != 0) != (torch.any(a != 0, dim=2))).any())
