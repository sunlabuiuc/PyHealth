# import argparse
# import torch
# import numpy as np
# import torch.nn as nn
# from torch import optim
# from torch.utils.data import DataLoader
# from collections import Counter
# import pickle
# from tqdm import tqdm
# from datetime import datetime
# from model import VariationalGNN
# from utils import train, evaluate, EHRData, collate_fn
# import os
# import logging
# from pyhealth.datasets import MIMIC3Dataset
# from sklearn.model_selection import train_test_split

# #configuration parameters
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--n_epochs', type=int, default=100)
# parser.add_argument('--in_feature', type=int, default=128)
# parser.add_argument('--out_feature', type=int, default=128)
# parser.add_argument('--n_heads', type=int, default=4)
# parser.add_argument('--n_layers', type=int, default=2)
# parser.add_argument('--dropout', type=float, default=0.3)
# parser.add_argument('--alpha', type=float, default=0.2)
# parser.add_argument('--reg', type=bool, default=True)
# parser.add_argument('--lbd', type=float, default=0.1)
# parser.add_argument('--eval_freq', type=int, default=100)
# parser.add_argument('--result_root', type=str, default='./results')
# args = parser.parse_args()

# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
# # print(device)

# def main():
#     #preprocessing data format w/ PyHealth
#     mimic_df = MIMIC3Dataset(
#         root="Final_MIMIC-III",
#         tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "LABEVENTS"],
#         dev=False,
#         refresh_cache=True,
#     )
#     print(mimic_df.info())


#     #initialize model with 1 node
#     initial_model = VariationalGNN(
#         in_features=args.in_feature,
#         out_features=args.out_feature,
#         num_of_nodes=1,
#         n_heads=args.n_heads,
#         n_layers=args.n_layers,
#         dropout=args.dropout,
#         alpha=args.alpha,
#         variational=args.reg,
#         none_graph_features=0
#     )

#     #convert data to features used and prediction required (x and y)
#     X, y = initial_model.convert_to_model_format(mimic_df)

#     #split data into train, validation, and test sets
#     X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
#     X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

#     #update model with necessary # of nodes
#     num_of_nodes = X_train.shape[1] + 1
#     model = VariationalGNN(
#         in_features=args.in_feature,
#         out_features=args.out_feature,
#         num_of_nodes=num_of_nodes,
#         n_heads=args.n_heads,
#         n_layers=args.n_layers,
#         dropout=args.dropout,
#         alpha=args.alpha,
#         variational=args.reg,
#         none_graph_features=0
#     ).to(device)

#     #check for multiple GPUs -> not needed for personal devices
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)

#     #data loader for training dataset
#     train_loader = DataLoader(
#         dataset=EHRData(X_train, y_train),
#         batch_size=args.batch_size,
#         collate_fn=collate_fn,
#         num_workers=torch.cuda.device_count(),
#         shuffle=True
#     )

#     #data loader for validation dataset
#     val_loader = DataLoader(
#         dataset=EHRData(X_val, y_val),
#         batch_size=args.batch_size,
#         collate_fn=collate_fn,
#         num_workers=torch.cuda.device_count(),
#         shuffle=False
#     )

#     #optimizer w/ regularization of 1e-8 applied to sum of weights
#     optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], 
#                           lr=args.lr, weight_decay=1e-8)

#     #scheduler to increment learning by 0.5 every 5 epochs
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#     #check for the shape of the input and target data -> check for correct format
#     print("Input shape:", next(iter(train_loader))[0].shape)
#     print("Target shape:", next(iter(train_loader))[1].shape)
#     for i, batch in enumerate(train_loader):
#         x, y, *other = batch
#         if i == 0:
#             print("Sample input range:", torch.min(x).item(), torch.max(x).item())
#             print("Sample target range:", torch.min(y).item(), torch.max(y).item())
#             break

#     #training
#     for epoch in range(args.n_epochs):
#         print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
#         #gets the ratio of positive and negative samples
#         ratio = Counter(y_train)
#         #calculates the pos_weight for the loss function
#         pos_weight = torch.ones(1).float().to(device) * (ratio[False] / ratio[True])
#         #loss function
#         criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        
#         #sets model to training and hold tracking for different types oflosses
#         t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
#         model.train()
#         total_loss = np.zeros(3)
        
#         #training loop through all the batches in the epoch
#         for idx, batch_data in enumerate(t):
#             #trains step and calculates loss
#             loss, kld, bce = train(batch_data, model, optimizer, criterion, args.lbd, 5)
#             #adds the different types of losses to total loss
#             total_loss += np.array([loss, bce, kld])
            
#             #saves the model parameters and evaluates model
#             if idx % args.eval_freq == 0 and idx > 0:
#                 torch.save(model.state_dict(), 
#                          f"{args.result_root}/parameter_{epoch}_{idx}")
#                 val_auprc, _ = evaluate(model, val_loader, len(y_val))
#                 log_msg = ('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
#                           (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
#                 logging.info(log_msg)
#                 print(log_msg)
                
#             if idx % 50 == 0 and idx > 0:
#                 t.set_description('[epoch:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
#                                 (epoch + 1, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
#                 t.refresh()

#          #adjusts the learning rate at each loop       
#         scheduler.step()

# if __name__ == '__main__':
#     main()
