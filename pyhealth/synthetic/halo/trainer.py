import os
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

import logging

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for the HALO synthetic data generator module.

    Args:
        dataset: PyHealth BaseEHRDataset which training will be conducted on; does not need to be a training subset
        model: a HALO model to train
        optimizer: pytorch optimizer
        checkpoint_dir: dir for writing checkpoint data to
        model_save_name: name of file within `checkpoint_dir` for writing model checkpoints to; model checkpoints are a dictionary with keys `model`, `iterations`, `epoch`, `optimizer`
        device: training device
    """
    def __init__(self, 
            dataset: datasets.BaseEHRDataset, 
            model: nn.Module,
            processor: Processor, 
            optimizer: Optimizer,
            checkpoint_dir: str,
            model_save_name: str,
            folds: int = 5
        ) -> None:
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model_save_name = model_save_name
        self.folds = folds
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_model_checkpoint_path(self):
        """Return `checkpoint_dir/model_save_name.pt`"""
        return os.path.join(self.checkpoint_dir, f"{self.model_save_name}.pt")
    
    def set_basic_splits(self, from_save=False, save=True):
        """Set class variables of a 0.8, 0.1, 0.1 split for train, test, eval split. 

            If `from_save` is True, will attempt to load each split from <halo.Trainer.checkpoint_dir>/<split_name>.pt. Upon failure, will create new splits with the underlying dataset. 
            If `save` is True, will store the set splits in memory at path <halo.Trainer.checkpoint_dir>/<split_name>.pt

            Args
        
            Returns: train, test, eval split
        """
        if from_save:
            try:
                self.train_dataset = torch.load(open(f"{self.checkpoint_dir}/train_dataset.pt", 'rb'))
                self.test_dataset = torch.load(open(f"{self.checkpoint_dir}/test_dataset.pt", 'rb'))
                self.eval_dataset = torch.load(open(f"{self.checkpoint_dir}/eval_dataset.pt", 'rb'))
                
                return self.train_dataset, self.test_dataset, self.eval_dataset
            except:
                logger.debug("failed to load basic splits from memory, generating splits from source dataset.")
        
        train, test, eval = self.split()
        self.train_dataset = train
        self.test_dataset = test
        self.eval_dataset = eval

        if save:
            torch.save(self.train_dataset, open(f"{self.checkpoint_dir}/train_dataset.pt", 'wb'))
            torch.save(self.test_dataset, open(f"{self.checkpoint_dir}/test_dataset.pt", 'wb'))
            torch.save(self.eval_dataset, open(f"{self.checkpoint_dir}/eval_dataset.pt", 'wb'))
        
        return self.train_dataset, self.test_dataset, self.eval_dataset
    
    def set_fold_splits(self, from_save=False, save=True):
        """Set class variables of a 0.8, 0.1, 0.1 split for train, test, eval split for each fold in a k-fold setup. 

            If `from_save` is True, will attempt to load each split from <halo.Trainer.checkpoint_dir>/<split_name>_<fold_number>.pt. Upon failure, will create new splits with the underlying dataset. 
            If `save` is True, will store the set splits in memory at path <halo.Trainer.checkpoint_dir>/<split_name>_<fold_number>.pt

            Args
        
            Returns: train, test, eval split
        """
        offset_per_fold = 1.0 / self.folds
        for fold in range(self.folds):
            if from_save:
                try:
                    self.train_dataset = torch.load(open(f"{self.checkpoint_dir}/train_dataset_{fold}.pt", 'rb'))
                    self.test_dataset = torch.load(open(f"{self.checkpoint_dir}/test_dataset_{fold}.pt", 'rb'))
                    self.eval_dataset = torch.load(open(f"{self.checkpoint_dir}/eval_dataset_{fold}.pt", 'rb'))    
                    loaded = True
                except:
                    loaded = False
                    logger.debug(f"failed to load basic splits from memory, generating splits from source dataset for fold {fold}.")
            if not from_save or not loaded:
                train, test, eval = self.split(shuffle=False, fold_offset=offset_per_fold * fold)
                self.train_dataset = train
                self.test_dataset = test
                self.eval_dataset = eval
                if save:
                    torch.save(self.train_dataset, open(f"{self.checkpoint_dir}/train_dataset_{fold}.pt", 'wb'))
                    torch.save(self.test_dataset, open(f"{self.checkpoint_dir}/test_dataset_{fold}.pt", 'wb'))
                    torch.save(self.eval_dataset, open(f"{self.checkpoint_dir}/eval_dataset_{fold}.pt", 'wb'))
    
    def load_fold_split(self, fold, from_save=False, save=True):
        """Set class variables of a 0.8, 0.1, 0.1 split for train, test, eval split for each fold in a k-fold setup. 

            If `from_save` is True, will attempt to load each split from <halo.Trainer.checkpoint_dir>/<split_name>_<fold_number>.pt. Upon failure, will create new splits with the underlying dataset. 
            If `save` is True, will store the set splits in memory at path <halo.Trainer.checkpoint_dir>/<split_name>_<fold_number>.pt

            Args
        
            Returns: train, test, eval split
        """
        offset_per_fold = 1.0 / self.folds
        if from_save:
            try:
                self.train_dataset = torch.load(open(f"{self.checkpoint_dir}/train_dataset_{fold}.pt", 'rb'))
                self.test_dataset = torch.load(open(f"{self.checkpoint_dir}/test_dataset_{fold}.pt", 'rb'))
                self.eval_dataset = torch.load(open(f"{self.checkpoint_dir}/eval_dataset_{fold}.pt", 'rb'))    
                
                return self.train_dataset, self.test_dataset, self.eval_dataset
            except:
                logger.debug(f"failed to load basic splits from memory, generating splits from source dataset for fold {fold}.")
                
        train, test, eval = self.split(shuffle=False, fold_offset=offset_per_fold * fold)
        self.train_dataset = train
        self.test_dataset = test
        self.eval_dataset = eval
        if save:
            torch.save(self.train_dataset, open(f"{self.checkpoint_dir}/train_dataset_{fold}.pt", 'wb'))
            torch.save(self.test_dataset, open(f"{self.checkpoint_dir}/test_dataset_{fold}.pt", 'wb'))
            torch.save(self.eval_dataset, open(f"{self.checkpoint_dir}/eval_dataset_{fold}.pt", 'wb'))
            
        return self.train_dataset, self.test_dataset, self.eval_dataset
        
    def split(self, splits: List[float] = [0.8, 0.1, 0.1], shuffle: bool = True, fold_offset: float = 0.0):
        """Split the dataset by ratio & return the result

        Args:
            splits: A list of ratios denoting portion of the dataset per split
            shuffle: whether to shuffle the dataset randomly prior to splitting. 
            fold_offset: offset used to rotate the dataset for k-fold cross validation.

        Returns:
            the computed dataset splits.
        """
        if shuffle:
            random.shuffle(self.dataset.patient_ids)
            
        if sum(splits) != 1:
            raise Exception(f"splits don't sum to the full dataset. sum(splits) = {sum(splits)}")
        
        n = len(self.dataset.patients)
        dataset_splits = []
        start_offset = int(n * fold_offset)
        for s in splits:
            n_split = int(n * s) # size of the current split

            # For a wrap-around effect, we'll use modulo operation
            end_offset = (start_offset + n_split) % n
            
            if start_offset < end_offset:
                subset = self.dataset[start_offset: end_offset]
            else:
                # wrap-around: combining the end and start portions
                subset = self.dataset[start_offset:] + self.dataset[:end_offset]

            dataset_splits.append(subset)
            start_offset = end_offset
            
        return dataset_splits
    
    def make_checkpoint(self, epoch, iteration):
        """Make a training checkpoint, recording `model`, `optimizer`, `iteration`, `epoch` in a dict at 
        `Trainer.checkpoint_dir` passed in during initialization. Uses `torch.save`
        
        Args:
            epoch: epoch to store
            iterations: iterations to store
        """
        state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': iteration,
                'epoch': epoch
            }

        torch.save(state, open(self.get_model_checkpoint_path(), 'wb'))
        print('\n------------ Save best model ------------\n')

    def eval(self, batch_size: int):
        """Compute current current loss on `Trainer.eval_dataset`, averaged across batches.
        
        Args:
            batch_size: batch size to use during evaluation

        Returns:
            average loss on `Trainer.eval_dataset` averaged over batches.
        """
        self.model.eval()
        
        with torch.no_grad():
            
            val_l = []
            for batch_ehr, batch_mask in self.processor.get_batch(self.eval_dataset, batch_size):
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)

                val_loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                val_l.append((val_loss).cpu().detach().numpy())
                
            return np.mean(val_l)

    def train(self, batch_size: int, epoch: int, patience: int, eval_period: int) -> None:
        """Conduct training on pyhealth.synthetic.halo module.

        Args:
            batch_size: the batch size to train with
            epoch: number of epochs to train for
            patience: number of eval_periods where loss is increasing before early termination of training
            eval_period: number of batches to train on before conducting evaluation on `Trainer.eval_dataset`. 
                If number provided exceeds the number of batches which exist in `Trainer.eval_dataset`,
                will perform evaluation after all batches in the epoch are trained. 
        
        """
        
        global_val_loss = 1e10
        current_patience = 0
        for e in tqdm(range(epoch), desc="Training HALO model"):
            self.model.train()
            random.shuffle(self.train_dataset)
            train_losses = []
            for i, (batch_ehr, batch_mask) in enumerate(self.processor.get_batch(self.train_dataset, batch_size)):
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                
                loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                
                loss.backward()
                self.optimizer.step()
                train_losses.append((loss).cpu().detach().numpy())
                
                # the eval period may never be reached if there aren't enough batches
                if i % min(eval_period, len(self.train_dataset)//batch_size - 1) == 0:
                    cur_train_loss = np.mean(train_losses)
                    train_losses = []
                    print("Epoch %d, Iter %d: Training Loss:%.7f"%(e, i, cur_train_loss))
                    cur_val_loss = self.eval(batch_size=batch_size)
                    print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))

                    # make checkpoint if our validation set is improving
                    if cur_val_loss < global_val_loss:
                        global_val_loss = cur_val_loss
                        current_patience = 0
                        self.make_checkpoint(epoch=e, iteration=i)
                    else:
                        current_patience += 1

            if current_patience >= patience: 
                print("Training parameter `patience` exceeded provided threshold.")
                break