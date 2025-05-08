#!/usr/bin/env python3
# model.py
import os
import time

import numpy
import pandas
import torch
import torchvision

import sklearn

from torch.nn import Module
import torch.nn.functional as F

from tqdm import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))

class Adversary(Module):
    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))
        
class CXRClassifier(object):
    'A classifier for various pathologies found in chest radiographs'
    def __init__(self):
        '''
        Create a classifier for chest radiograph pathology.
        '''
        self.lossfunc = torch.nn.BCELoss()

    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=100, 
              lr=0.01, 
              weight_decay=1e-4,
              batch_size=16,
              early_stopping_rounds=3,
              logpath=None,
              checkpoint_path='checkpoint.pkl',
              verbose=True,
              masked=None):
        '''
        Train the classifier to predict the labels in the specified dataset.
        Training will start from the weights in a densenet-121 model pretrained
        on imagenet, as provided by torchvision.
        
        Args:
            train_dataset: An instance of MIMICDataset or 
                CheXpertDataset. Used for training neural network.
            val_dataset: An instance of MIMICDataset or 
                CheXpertDataset. Used for determining when to stop training.
            max_epochs (int): The maximum number of epochs for which to train.
            lr (float): The learning rate to use during training.
            weight_decay (float): The weight decay to use during training.
            batch_size (int): The size of mini-batches to use while training.
            early_stopping_rounds (int): The number of rounds, in which the
                validation loss does not improve from its best value, to wait 
                before stopping training.
            logpath (str): The path at which to write a training log. If None,
                do not write a log.
            checkpoint_path (str): The path at which to save a checkpoint 
                corresponding to the model so far with the best validation loss.
            verbose (bool): If True, print extra information about training.
        Returns:
            model: Trained instance of torch.nn.Module.
        '''
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.weight_decay = weight_decay
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        # Build the model
        self.model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        # Add a classification head; consists of standard dense layer with
        # sigmoid activation and one output node per pathology in train_dataset
        self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, len(train_dataset.labels)), 
                torch.nn.Sigmoid())

        # Put model on GPU
        self.model.to(torch.device("cpu"))

        # Define the optimizer. Use SGD with momentum and weight decay.
        optimizer = self._get_optimizer(lr, self.weight_decay)
        best_loss = None 
        best_epoch = None 

        # Begin training. Iterate over each epoch to (i) optimize network and
        # (ii) calculate validation loss.
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                # Iterate over each batch of data; loss holds a 
                # running sum of the loss from each batch
                loss = 0
                for batch in tqdm(dataloaders[phase]):
                    inputs, labels, _, ds = batch
                    # batch size may differ from batch_size for the last  
                    # batch in an epoch
                    current_batch_size = inputs.shape[0]

                    # Transfer inputs (images) and labels (arrays of ints) to 
                    # GPU
                    inputs = torch.autograd.Variable(inputs.to(device))
                    labels = torch.autograd.Variable(labels.to(device)).float()
                    if masked is not None:
                        ds = torch.autograd.Variable(ds.to(device)).float()

                    outputs = self.model(inputs)

                    # Calculate the loss
                    optimizer.zero_grad()
                    batch_loss = self.lossfunc(outputs, labels)

                    # If training, update the network's weights
                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

                    # Update the running sum of the loss
                    loss += batch_loss.data.item()*current_batch_size
                dataset_size = len(val_dataset) if phase == 'val' else len(train_dataset)
                loss /= dataset_size
                if phase == 'train':
                    trainloss = loss
                if phase == 'val':
                    valloss = loss

                if phase == 'val':
                    # Check if the validation loss is the best we have seen so
                    # far. If so, record that epoch.
                    if best_loss is None or valloss < best_loss:
                        best_epoch = i_epoch
                        best_loss = valloss
                        self._checkpoint(i_epoch, valloss)
                    # If the validation loss has not improved, decay the 
                    # learning rate
                    else:
                        self.lr /= 10
                        optimizer = self._get_optimizer(
                                 self.lr, 
                                 self.weight_decay)

            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)

            # If we have gone three epochs without improvement in the validation
            # loss, stop training
            if i_epoch - best_epoch > early_stopping_rounds:
                break
        self.load_checkpoint(self.checkpoint_path)
        return self.model

    def _checkpoint(self, epoch, valloss):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'model': self.model,
                 'best_loss': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr }
        torch.save(state, self.checkpoint_path)

    def _get_optimizer(self, lr, weight_decay):
        opt = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay)
        return opt

    def load_checkpoint(self, path):
        self.model = torch.load(path, weights_only=False)['model']

    def predict(self, dataset, batch_size=16):
        '''
        Predict the labels of the images in 'dataset'. Outputs indicate the
        probability of a particular label being positive (interpretation 
        depends on the dataset used during training).

        Args:
            dataset: An instance of MIMICDataset or 
                CheXpertDataset.
        Returns:
            predictions (numpy.ndarray): An array of floats, of shape 
                (nsamples, nlabels), where predictions[i,j] indicates the 
                probability of label j of sample i being positive.
        '''
        self.model.train(False)

        # Build a dataloader to batch predictions
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        pred_df = pandas.DataFrame(columns=["path"])
        true_df = pandas.DataFrame(columns=["path"])

        output = numpy.zeros((len(dataset), len(dataset.labels)))

        # Iterate over the batches
        for ibatch, batch in enumerate(dataloader):
            inputs, labels, _, ds = batch
            # Move to GPU
            inputs = torch.autograd.Variable(inputs.to(device))
            labels = torch.autograd.Variable(labels.to(device))

            true_labels = labels.cpu().data.numpy()
            # Size of current batch. Could be less than batch_size in final 
            # batch
            current_batch_size = true_labels.shape[0]

            # perform prediction
            probs = self.model(inputs).cpu().data.numpy()

            # get predictions and true values for each item in batch
            for isample in range(0, current_batch_size):
                for ilabel in range(len(dataset.labels)):
                    output[batch_size*ibatch + isample, ilabel] = \
                        probs[isample, ilabel]
        return output
        
class CXRAdvClassifier(object):
    '''A CXR classifier f(X) that is independent of AP/PA View.
    Based on "Learning to Pivot with Adversarial Networks" (Louppe et al. 2016). 
    Code also inspired by PyTorch implementation found on
    https://github.com/equialgo/fairness-in-ml/blob/master/fairness-in-torch.ipynb'''
    def __init__(self):
        '''
        Create a classifier for chest radiograph pathology.
        '''
        self.lossfunc = torch.nn.BCELoss()
    
    def predict(self, dataset, batch_size=16):
        '''
        Predict the labels of the images in 'dataset'. Outputs indicate the
        probability of a particular label being positive (interpretation 
        depends on the dataset used during training).

        Args:
            dataset: An instance of a CXRDataset.
        Returns:
            predictions (numpy.ndarray): An array of floats, of shape 
                (nsamples, nlabels), where predictions[i,j] indicates the 
                probability of label j of sample i being positive.
        '''
        self.clf.train(False)

        # Build a dataloader to batch predictions
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        pred_df = pandas.DataFrame(columns=["path"])
        true_df = pandas.DataFrame(columns=["path"])

        output = numpy.zeros((len(dataset), len(dataset.labels)))

        # Iterate over the batches
        for ibatch, batch in enumerate(dataloader):
            inputs, labels, _, appa = batch
            # Move to GPUls
            inputs = torch.autograd.Variable(inputs.to(device))
            labels = torch.autograd.Variable(labels.to(device))

            true_labels = labels.cpu().data.numpy()
            # Size of current batch. Could be less than batch_size in final 
            # batch
            current_batch_size = true_labels.shape[0]

            # perform prediction
            probs = self.clf(inputs).cpu().data.numpy()

            # get predictions and true values for each item in batch
            for isample in range(0, current_batch_size):
                for ilabel in range(len(dataset.labels)):
                    output[batch_size*ibatch + isample, ilabel] = \
                        probs[isample, ilabel]
        return output
    
    def predict_nuisance(self, dataset, score, batch_size=16):
        '''
        Adversary's predictions for nuisance/protected class using
        the score of the classifier as input

        Args:
            dataset: An instance of a CXRDataset.
        Returns:
            predictions (numpy.ndarray): An array of floats, of shape 
                (nsamples, nlabels), where predictions[i,j] indicates the 
                probability of label j of sample i being positive.
        '''
        self.clf.train(False)
        score = torch.tensor(score)
        score = torch.autograd.Variable(score).to(device).float().view(-1,1)
        output = self.adv(score).cpu().detach().numpy()
        
        z_labels = numpy.zeros(len(dataset))
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        # Iterate over the batches
        for ibatch, batch in enumerate(dataloader):
            inputs, labels, _, appa = batch
            # Move to GPU
            inputs = torch.autograd.Variable(inputs.to(device))
            labels = torch.autograd.Variable(labels.to(device))
            appa_numeric = [1.0 if x == 'PA' else 0.0 for x in appa]
            appa = torch.tensor(appa_numeric, dtype=torch.float32).to(device).view(-1, 1)
            
            true_z = appa.cpu().numpy()
            # Size of current batch. Could be less than batch_size in final 
            # batch
            current_batch_size = true_z.shape[0]
            
            for isample in range(0, current_batch_size):
                z_labels[batch_size*ibatch + isample] = true_z[isample]
        
        
        return output, z_labels
    
    def _pretrain_adversary(self,
                            train_dataloader,
                            optimizer,
                            criterion,
                            pneumo_index,
                            lam = 1.):
        self.clf.train(False)
        for image, label, _, appa in tqdm(train_dataloader):
            image = torch.autograd.Variable(image).to(device)
            label = torch.autograd.Variable(label).to(device)
            appa_numeric = [1.0 if x == 'PA' else 0.0 for x in appa]
            appa = torch.tensor(appa_numeric, dtype=torch.float32).to(device).view(-1, 1)


            p_y = self.clf(image).detach()
            p_y_pneumo = p_y[:,pneumo_index].view(-1,1)
            self.adv.zero_grad()
            p_z = self.adv(p_y_pneumo)
            appa = appa.expand_as(p_z)
            loss = (criterion(p_z, appa) * lam).mean()
            loss.backward()
            optimizer.step()
            
        return self.adv
    
    def _pretrain_classifier(self, 
                             train_dataset,
                             optimizer,
                             clf_pretrain_epochs=7,
                             batch_size=16):

        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8)
        
        for i_epoch in range(clf_pretrain_epochs):
            print("-- CLF Pretrain Epoch {:03d} --".format(i_epoch))
            
            self.clf.train(True)
                
            for batch in tqdm(train_dataloader):
                inputs, labels, _, appa = batch
                # batch size may differ from batch_size for the last  
                # batch in an epoch
                current_batch_size = inputs.shape[0]

                # Transfer inputs (images) and labels (arrays of ints) to 
                # GPU
                inputs = torch.autograd.Variable(inputs.to(device))
                labels = torch.autograd.Variable(labels.to(device)).float()
                # Convert 'AP' → 0 and 'PA' → 1 (or any mapping you prefer)
                appa_numeric = [1.0 if x == 'PA' else 0.0 for x in appa]
                appa = torch.tensor(appa_numeric, dtype=torch.float32).to(device).view(-1, 1)

                outputs = self.clf(inputs)

                # Calculate the loss
                optimizer.zero_grad()
                batch_loss = self.lossfunc(outputs, labels)
                batch_loss.backward()
                optimizer.step()

        return self.clf
    
    def _batch_adversarial_train(self,
                                 train_dataloader,
                                 clf_criterion,
                                 adv_criterion,
                                 clf_optimizer,
                                 adv_optimizer,
                                 pneumo_index,
                                 lam = 1.):
        # Train adversary
        self.clf.train(False)
        for image, label, _, appa in tqdm(train_dataloader):
            
            image = torch.autograd.Variable(image).to(device)
            label = torch.autograd.Variable(label).to(device)
            appa_numeric = [1.0 if x == 'PA' else 0.0 for x in appa]
            appa = torch.tensor(appa_numeric, dtype=torch.float32).to(device).view(-1, 1)
            
            p_y = self.clf(image)
            p_y_pneumo = p_y[:,pneumo_index].view(-1,1)
            self.adv.zero_grad()
            p_z = self.adv(p_y_pneumo)
            appa = appa.expand_as(p_z)  # Match shape with p_z
            loss_adv = (adv_criterion(p_z, appa) * lam).mean()
            loss_adv.backward()
            adv_optimizer.step()
            
        # Train classifier on single batch
        self.clf.train(True)
        for image, label, _, appa in train_dataloader:
            pass
        image = torch.autograd.Variable(image).to(device)
        label = torch.autograd.Variable(label).to(device).float()
        appa_numeric = [1.0 if x == 'PA' else 0.0 for x in appa]
        appa = torch.tensor(appa_numeric, dtype=torch.float32).to(device).view(-1, 1)
        p_y = self.clf(image)
        p_y_pneumo = p_y[:,pneumo_index].view(-1,1)
        p_z = self.adv(p_y_pneumo)
        self.clf.zero_grad()
        p_z = self.adv(p_y_pneumo)
        appa = appa.expand_as(p_z)  # Match shape with p_z
        loss_adv = (adv_criterion(p_z, appa) * lam).mean()
        clf_loss = clf_criterion(p_y, label) - (adv_criterion(self.adv(p_y_pneumo), appa) * lam).mean()
        clf_loss.backward()
        clf_optimizer.step()
        
        return self.clf, self.adv, clf_loss, loss_adv

    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=500, 
              lr=0.01, 
              weight_decay=1e-4,
              batch_size=16,
              logpath=None,
              checkpoint_path='checkpoint.pkl',
              verbose=True,
              pretrained_classifier_path=None):
        '''
        Train the classifier to predict the labels in the specified dataset.
        Training will start from the weights in a densenet-121 model pretrained
        on imagenet, as provided by torchvision.
        
        Args:
            train_dataset: An instance of ChestXray14Dataset, MIMICDataset, or 
                CheXpertDataset. Used for training neural network.
            val_dataset: An instance of ChestXray14Dataset, MIMICDataset, or 
                CheXpertDataset. Used for determining when to stop training.
            max_epochs (int): The maximum number of epochs for which to train.
            lr (float): The learning rate to use during training.
            weight_decay (float): The weight decay to use during training.
            batch_size (int): The size of mini-batches to use while training.
            logpath (str): The path at which to write a training log. If None,
                do not write a log.
            checkpoint_path (str): The path at which to save a checkpoint 
                corresponding to the model so far with the best validation loss.
            verbose (bool): If True, print extra information about training.
            pretrained_classifier_path (str): The path at which a pretrained 
                standard CXRClassifier can be found
        Returns:
            model: Trained instance of torch.nn.Module.
        '''
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.weight_decay = weight_decay
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        # Build the classifier model
        # Note that pretraining has already been done on ImageNet
        self.clf = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.clf.classifier.in_features
        # Add a classification head; consists of standard dense layer with
        # sigmoid activation and one output node per pathology in train_dataset
        self.clf.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, len(train_dataset.labels)), 
                torch.nn.Sigmoid())
        # Define the optimizer. Use SGD with momentum and weight decay.
        clf_criterion = torch.nn.BCELoss()
        clf_optimizer = self._get_optimizer(lr, self.weight_decay)
        clf_optimizer_2 = self._get_optimizer(lr/10., self.weight_decay)

        # Put classifier on GPU
        self.clf.to(device)
        
        # Build and pretrain the adversary
        self.adv = Adversary(1)
        adv_criterion = torch.nn.BCELoss(reduce=False)
        adv_optimizer = torch.optim.Adam(self.adv.parameters())
        # Put adversary on GPU
        self.adv.to(device)
        
        pneumo_index = _find_index(train_dataset, 'pneumonia')
        
        if pretrained_classifier_path is not None:
            print('--Loading classifier from {}----'.format(pretrained_classifier_path))
            self.clf = torch.load(pretrained_classifier_path)['model']
        
        else:
            N_CLF_EPOCHS = 7
            print('--Pretraining classifier----')
            self._pretrain_classifier(train_dataset,clf_optimizer,
                                      clf_pretrain_epochs=N_CLF_EPOCHS)
        
        N_ADV_EPOCHS = 1
        print('--Pretraining adversary----')
        for pretrain_epoch in range(N_ADV_EPOCHS):
            print('--Pretrain epoch:{:d}----'.format(pretrain_epoch))
            self.adv = self._pretrain_adversary(train_dataloader,
                                                adv_optimizer,
                                                adv_criterion,
                                                pneumo_index,
                                                lam = 1.)
            
        best_loss = None 
        best_epoch = None
        best_val_auroc_clf = None

        # Begin training. Iterate over each epoch to (i) optimize network and
        # (ii) calculate validation loss.
        print('--Begin joint adverarial optimization----')
        for i_epoch in range(max_epochs):
            print("-------- Batch 'Epoch' {:03d} --------".format(i_epoch))
            
            self.clf, self.adv, clf_loss, loss_adv = self._batch_adversarial_train(train_dataloader,
                                                                                   clf_criterion, 
                                                                                   adv_criterion,
                                                                                   clf_optimizer_2, 
                                                                                   adv_optimizer,
                                                                                   pneumo_index,
                                                                                   lam = 1.)
            
            ## print training loss, val loss, training adv auroc, val adv auroc
            # get val loss here
            
            val_probs = self.predict(val_dataset)
            val_true = val_dataset.get_all_labels()
            valloss = self.lossfunc(torch.tensor(val_true),torch.tensor(val_probs))
            
            adv_val_probs, adv_val_true = self.predict_nuisance(val_dataset,val_probs[:,pneumo_index])
            
            val_auroc_adv = sklearn.metrics.roc_auc_score(
                            adv_val_true,
                            adv_val_probs)
            val_auroc_clf = sklearn.metrics.roc_auc_score(
                            val_true[:,pneumo_index],
                            val_probs[:,pneumo_index])
            
            logstr = "Epoch {:03d}: ".format(i_epoch)                \
                     + "train clf loss {:08.4f},".format(clf_loss)    \
                + "train adv loss {:08.4f},".format(loss_adv)    \
                    + "val clf loss {:08.4f},".format(valloss)       \
                        + "val_auroc_adv {:08.4f},".format(val_auroc_adv) \
                        + "val_auroc_clf {:08.4f}".format(val_auroc_clf)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')      
            print(logstr)
            
            if (val_auroc_adv <= 0.52 and val_auroc_adv >= 0.48) and (best_val_auroc_clf == None or val_auroc_clf > best_val_auroc_clf):
                best_epoch = i_epoch
                best_val_auroc_clf = val_auroc_clf
                self._checkpoint(i_epoch, val_auroc_clf)

        return self.clf, self.adv

    def _checkpoint(self, epoch, valloss):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'clf': self.clf,
                 'adv': self.adv,
                 'best_val_auroc_clf': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr }
        torch.save(state, self.checkpoint_path)

    def _get_optimizer(self, lr, weight_decay):
        opt = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.clf.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay)
        return opt

    def load_checkpoint(self, path):
        saved_dict = torch.load(path)
        self.clf = saved_dict['clf']
        self.adv = saved_dict['adv']