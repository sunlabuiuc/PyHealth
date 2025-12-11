import torch
from torch.utils.data import DataLoader
from model_demo import MiniViewXGen
from dataset_demo import dataset

model = MiniViewXGen()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for epoch in range(2):
    for batch in loader:
        tokens = torch.tensor(batch["image_tokens"])
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        out = model(inp)
        loss = loss_fn(out.reshape(-1, 256), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch", epoch, "Loss", float(loss))
