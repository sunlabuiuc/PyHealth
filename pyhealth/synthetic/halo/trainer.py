import random
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pyhealth import datasets
from pyhealth.synthetic.halo.processor import Processor

class Trainer:
    def __init__(self, 
            dataset: datasets, 
            model: nn.Module,
            processor: Processor, 
            optimizer: Optimizer,
            checkpoint_path: str,
        ) -> None:
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    """
    Set class variables of a 0.8, 0.1, 0.1 split for train, test, eval sets respectivley.
    returns the splits for convenience
    """
    def set_basic_splits(self):
        train, test, eval = self.split()
        self.train_dataset = train
        self.test_dataset = test
        self.eval_dataset = eval
        
        return self.train_dataset, self.test_dataset, self.eval_dataset
        
    def split(self, splits: List[float] = [0.8, 0.1, 0.1], shuffle: bool = False):
        if shuffle:
            self.dataset = random.random.shuffle(self.dataset)
            
        if sum(splits) != 1:
            raise Exception(f"splits don't sum to the full dataset. sum(splits) = {sum(splits)}")
        
        n = len(self.dataset.patients)
        dataset_splits = []
        start_offset = 0
        for s in splits:
            n_split = math.ceil(n * s) # size of the current split
            
            # the last subset will be the smallest
            subset = self.dataset[start_offset: min(start_offset + n_split, n)]
           
            dataset_splits.append(subset)
            start_offset += n_split
            
        return dataset_splits
    
    def make_checkpoint(self, iteration):
        state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': iteration
            }
        torch.save(state, f"{self.checkpoint_path}.pkl")
        print('\n------------ Save best model ------------\n')

    def eval(self, batch_size: int, current_epoch: int = 0, current_iteration: int = 0, patience: int = 0, save=True):
        self.model.eval()
        
        with torch.no_grad():
            
            global_loss = 1e10
            val_l = []
            current_patience = 0
            for batch_ehr, batch_mask in self.processor.get_batch(self.eval_dataset, batch_size):
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)

                val_loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                val_l.append((val_loss).cpu().detach().numpy())
                
                cur_val_loss = np.mean(val_l)
                if current_epoch:
                    print("Epoch %d Validation Loss:%.7f"%(current_epoch, cur_val_loss))
                else:
                    print("Validation Loss:%.7f"%(cur_val_loss))

                # make checkpoint
                if save and cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    patience = 0
                    self.make_checkpoint(iteration=current_iteration)
                
                current_patience += 1
                if current_patience == patience: break
    
    def validate(self, batch_size):
        self.eval(batch_size=batch_size, current_epoch=0, current_iteration=0, patience=None)

    def train(self, batch_size: int, epoch: int, patience: int, eval_period: int) -> None:        
        
        for e in tqdm(range(epoch), desc="Training HALO model"):
            
            self.model.train()
            
            for i, (batch_ehr, batch_mask) in enumerate(self.processor.get_batch(self.train_dataset, batch_size)):
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                
                loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                
                loss.backward()
                self.optimizer.step()
                
                # the eval period may never be reached if there aren't enough batches
                if i % min(eval_period, len(self.train_dataset)//batch_size) == 0:
                    print("Epoch %d, Iter %d: Training Loss:%.7f"%(e, i, loss))
                    self.eval(current_epoch=e, current_iteration=i, batch_size=batch_size)

        self.eval(batch_size=batch_size, current_epoch=epoch, current_iteration=-1, patience=patience)