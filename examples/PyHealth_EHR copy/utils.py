# import torch
# import numpy as np
# from sklearn.metrics import precision_recall_curve, auc
# from torch.utils.data import Dataset

# #overall directly used from original paper -> 'Variationally Regularized Graph-based Representation Learningor Electronic Health Records

# #check for GPU
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
# print(device)


# def train(data, model, optim, criterion, lbd, max_clip_norm=5):
#     model.train()
#     input = data[:, :-1].to(device)
#     label = data[:, -1].float().to(device)
#     model.train()
#     optim.zero_grad()
#     logits, kld = model(input)
#     logits = logits.squeeze(-1)
#     kld = kld.sum()
#     bce = criterion(logits, label)
#     loss = bce + lbd * kld
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
#     loss.backward()
#     optim.step()
#     return loss.item(), kld.item(), bce.item()


# def evaluate(model, data_iter, length):
#     model.eval()
#     y_pred = np.zeros(length)
#     y_true = np.zeros(length)
#     y_prob = np.zeros(length)
#     pointer = 0
#     for data in data_iter:
#         input = data[:, :-1].to(device)
#         label = data[:, -1]
#         batch_size = len(label)
#         probability, _ = model(input)
#         probability = torch.sigmoid(probability.squeeze(-1).detach())
#         predicted = probability > 0.5
#         y_true[pointer: pointer + batch_size] = label.numpy()
#         y_pred[pointer: pointer + batch_size] = predicted.cpu().numpy()
#         y_prob[pointer: pointer + batch_size] = probability.cpu().numpy()
#         pointer += batch_size
#     precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
#     return auc(recall, precision), (y_pred, y_prob, y_true)


# class EHRData(Dataset):
#     def __init__(self, data, cla):
#         self.data = data
#         self.cla = cla

#     def __len__(self):
#         return len(self.cla)

#     def __getitem__(self, idx):
#         return self.data[idx], self.cla[idx]


# def collate_fn(data):
#     # padding
#     data_list = []
#     for datum in data:
#         data_list.append(np.hstack((datum[0].toarray().ravel(), datum[1])))
#     return torch.from_numpy(np.array(data_list)).long()
