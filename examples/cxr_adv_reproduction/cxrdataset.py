#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import sklearn.model_selection
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class CXRDataset(Dataset):
    _transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    _transforms['test'] = _transforms['val']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path_to_images, self.df.index[idx])
        image = Image.open(image_path).convert('RGB')

        label = np.zeros(len(self.labels), dtype=int)
        for i, col in enumerate(self.labels):
            val = self.df.iloc[idx][col]
            if pd.notna(val) and int(val) > 0:
                label[i] = 1

        if self.transform:
            image = self.transform(image)

        return image, label, self.df.index[idx], ['None']

    def get_all_labels(self):
        ndim = len(self.labels)
        nsamples = len(self)
        output = np.zeros((nsamples, ndim))
        for i in range(nsamples):
            output[i] = self[i][1]
        return output

class NIHDataset(CXRDataset):
    def __init__(self, fold='train', random_state=42):
        label_path = './data/Data_Entry_2017.csv'
        image_dir = './data/images-224/'
        trainval_list = './data/train_val_list_NIH.txt'
        test_list = './data/test_list_NIH.txt'

        df = pd.read_csv(label_path)
        df = df.set_index('Image Index')

        # Define the 14 NIH disease labels
        all_labels = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia'
        ]

        # One-hot encode each label into a column
        for label in all_labels:
            df[label] = df['Finding Labels'].str.contains(label).astype(int)

        with open(trainval_list, 'r') as f:
            all_trainval = f.read().splitlines()

        if fold == 'test':
            with open(test_list, 'r') as f:
                image_list = f.read().splitlines()
        elif fold in ['train', 'val']:
            from sklearn.model_selection import train_test_split
            train_split, val_split = train_test_split(
                all_trainval, test_size=0.1, random_state=random_state)
            image_list = train_split if fold == 'train' else val_split
        else:
            raise ValueError("Invalid fold")

        # Filter images to those that exist on disk
        image_list = [img for img in image_list if os.path.exists(os.path.join(image_dir, img))]

        self.df = df.loc[df.index.intersection(image_list)]
        self.labels = all_labels
        self.path_to_images = image_dir
        self.transform = self._transforms['train' if fold == 'train' else 'val']
