import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from preprocess.transforms import get_transforms_pretrain, get_transforms_classification

from pyhealth.datasets import SampleBaseDataset


class MedicalImageTextPairDataset(SampleBaseDataset):
    def __init__(
            self,
            data_path,
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            transform=None,
            dataset_type='chest',
            max_length=128,
            split="train"
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.split = split

        raw_samples = self._load_raw_data(data_path, dataset_type, split)

        super(MedicalImageTextPairDataset, self).__init__(
            samples=raw_samples,
            dataset_name=f"{dataset_type}-dataset",
            task_name="image-text-contrastive"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform = transform

    def _load_raw_data(self, data_path, dataset_type, split):
        pairs = []

        std_path = os.path.join(data_path, f"{dataset_type}_{split}_pairs.json")
        alt_path = os.path.join(data_path, dataset_type, split, f"{dataset_type}_pairs.json")

        if os.path.exists(std_path):
            with open(std_path, 'r') as f:
                raw_data = json.load(f)
                print(f"Found {len(raw_data)} pairs from {std_path}")

                for i, item in enumerate(raw_data):
                    if 'study_id' in item:
                        p_id = item['study_id']
                        v_id = item['study_id']
                    else:
                        img_file = item['image_path']
                        base_name = os.path.basename(img_file)
                        parts = os.path.splitext(base_name)[0].split('_')

                        if len(parts) > 1:
                            p_id = parts[0]
                            v_id = f"{parts[0]}_{parts[1]}"
                        else:
                            p_id = f"patient_{i}"
                            v_id = f"visit_{i}"

                    img_path = item['image_path']
                    if not os.path.isabs(img_path):
                        alt_img_path = os.path.join(os.path.dirname(alt_path), img_path)
                        if os.path.exists(alt_img_path):
                            img_path = alt_img_path

                    sample = {
                        'patient_id': p_id,
                        'visit_id': v_id,
                        'image_path': img_path,
                        'report': item['report']
                    }
                    pairs.append(sample)

        elif os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                raw_data = json.load(f)
                print(f"Found {len(raw_data)} pairs from alternate path: {alt_path}")

                for i, item in enumerate(raw_data):
                    if 'study_id' in item:
                        p_id = item['study_id']
                        v_id = item['study_id']
                    else:
                        img_file = item['image_path']
                        base_name = os.path.basename(img_file)
                        parts = os.path.splitext(base_name)[0].split('_')

                        if len(parts) > 1:
                            p_id = parts[0]
                            v_id = f"{parts[0]}_{parts[1]}"
                        else:
                            p_id = f"patient_{i}"
                            v_id = f"visit_{i}"

                    img_path = item['image_path']
                    if not os.path.isabs(img_path):
                        test_path = os.path.join(os.path.dirname(std_path), img_path)
                        if os.path.exists(test_path):
                            img_path = test_path

                    sample = {
                        'patient_id': p_id,
                        'visit_id': v_id,
                        'image_path': img_path,
                        'report': item['report']
                    }
                    pairs.append(sample)

        else:
            img_dir = os.path.join(data_path, dataset_type, split, 'images')
            rep_dir = os.path.join(data_path, dataset_type, split, 'reports')

            if not os.path.exists(img_dir) or not os.path.exists(rep_dir):
                img_dir = os.path.join(data_path, split, 'images')
                rep_dir = os.path.join(data_path, split, 'reports')

                if not os.path.exists(img_dir) or not os.path.exists(rep_dir):
                    raise FileNotFoundError(f"Can't find data in {data_path}")

            cnt = 0
            for i, img_fname in enumerate(os.listdir(img_dir)):
                if not img_fname.endswith(('.jpg', '.png', '.jpeg', '.dcm')):
                    continue

                img_id = os.path.splitext(img_fname)[0]

                report_fname = os.path.join(rep_dir, f"{img_id}.txt")
                if os.path.exists(report_fname):
                    try:
                        with open(report_fname, 'r') as f:
                            txt = f.read().strip()

                        parts = img_id.split('_')
                        if len(parts) > 1:
                            p_id = parts[0]
                            v_id = f"{parts[0]}_{parts[1]}"
                        else:
                            p_id = f"patient_{i}"
                            v_id = f"visit_{i}"

                        sample = {
                            'patient_id': p_id,
                            'visit_id': v_id,
                            'image_path': os.path.join(img_dir, img_fname),
                            'report': txt
                        }
                        pairs.append(sample)
                        cnt += 1
                    except IOError:
                        continue

            if cnt > 0:
                print(f"Loaded {cnt} pairs from directory structure")

        return pairs

    def _sample_text_span(self, report):
        import re
        sentences = re.split(r'[.!?]', report)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return report

        return random.choice(sentences)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = sample['image_path']
        if not os.path.isabs(img_path):
            paths_to_try = [
                os.path.join(self.data_path, img_path),
                os.path.join(self.data_path, self.dataset_type, self.split, img_path),
                os.path.join(self.data_path, self.split, img_path)
            ]

            for p in paths_to_try:
                if os.path.exists(p):
                    img_path = p
                    break

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        except IOError:
            print(f"Cannot read image: {img_path}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        txt = self._sample_text_span(sample['report'])
        tokens = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'images': img,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }


class ClassificationDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        if img_path.endswith('.dcm'):
            try:
                dcm = pydicom.dcmread(img_path)
                img_arr = dcm.pixel_array

                if img_arr.max() > 0:
                    img_arr = (img_arr / img_arr.max() * 255).astype(np.uint8)

                if len(img_arr.shape) == 2:
                    img_arr = np.stack([img_arr] * 3, axis=2)

                img = Image.fromarray(img_arr)
            except pydicom.errors.InvalidDicomError:
                print(f"Invalid DICOM file: {img_path}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            except AttributeError:
                print(f"Missing pixel data in DICOM: {img_path}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"Image not found: {img_path}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            except IOError:
                print(f"Cannot read image: {img_path}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


class RetrievalDataset(Dataset):
    def __init__(self, query_data, candidate_data, transform=None, is_text_query=False, tokenizer=None):
        self.query_data = query_data
        self.candidate_data = candidate_data
        self.transform = transform
        self.is_text_query = is_text_query
        self.tokenizer = tokenizer

        if is_text_query and tokenizer is None:
            raise ValueError("Text queries require a tokenizer")

    def __len__(self):
        return len(self.query_data)

    def get_query(self, idx):
        if self.is_text_query:
            txt = self.query_data[idx]['text']

            enc = self.tokenizer(
                txt,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )

            return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)
        else:
            img_path = self.query_data[idx]['image_path']
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img
            except FileNotFoundError:
                print(f"Query image not found: {img_path}")
                if self.transform:
                    return torch.zeros(3, 224, 224)
                else:
                    return Image.new('RGB', (224, 224), (0, 0, 0))
            except IOError:
                print(f"Cannot read query image: {img_path}")
                if self.transform:
                    return torch.zeros(3, 224, 224)
                else:
                    return Image.new('RGB', (224, 224), (0, 0, 0))

    def get_candidates(self):
        images = []
        labels = []

        for item in self.candidate_data:
            path = item['image_path']
            lbl = item['label']

            try:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)

                images.append(img)
                labels.append(lbl)
            except FileNotFoundError:
                print(f"Candidate image not found: {path}")
                continue
            except IOError:
                print(f"Cannot read candidate image: {path}")
                continue

        return torch.stack(images), labels

    def __getitem__(self, idx):
        query = self.get_query(idx)
        candidates, candidate_labels = self.get_candidates()

        query_label = self.query_data[idx]['label']

        return query, candidates, query_label, candidate_labels


def load_classification_dataset(logger, data_path, task_name, percent=100, mode='linear', batch_size=64):
    train_tfm, val_tfm = get_transforms_classification()
    if mode == 'linear':
        train_tfm = val_tfm

    if task_name == 'rsna':
        from preprocess.rsna import preprocess_rsna
        data = preprocess_rsna(
            logger=logger,
            data_path=data_path,
            train_transform=train_tfm,
            val_transform=val_tfm,
            percent=percent,
            batch_size=batch_size)
    elif task_name == 'chexpert':
        from preprocess.chexpert import preprocess_chexpert
        data = preprocess_chexpert(
            logger=logger,
            data_path=data_path,
            train_transform=train_tfm,
            val_transform=val_tfm,
            percent=percent,
            batch_size=batch_size)
    elif task_name == 'covidx':
        from preprocess.covidx import preprocess_covidx
        data = preprocess_covidx(
            logger=logger,
            data_path=data_path,
            train_transform=train_tfm,
            val_transform=val_tfm,
            percent=percent,
            batch_size=batch_size)
    elif task_name == 'mura':
        from preprocess.mura import preprocess_mura
        data = preprocess_mura(
            logger=logger,
            data_path=data_path,
            train_transform=train_tfm,
            val_transform=val_tfm,
            percent=percent,
            batch_size=batch_size)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # According to paper: batch size should be 64
    train_loader = DataLoader(data['train_dataset'], batch_size=64, shuffle=True, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(data['val_dataset'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(data['test_dataset'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': data['num_classes']
    }


def load_retrieval_dataset(eval_data_dir, img_base_dir, task_name):
    # Retrieval follows the same transform as pretraining validation
    _, val_tfm = get_transforms_pretrain()

    if task_name == 'retrieval_img':
        q_csv = os.path.join(eval_data_dir, 'image-retrieval/query.csv')
        cand_csv = os.path.join(eval_data_dir, 'image-retrieval/candidate.csv')

        q_df = pd.read_csv(q_csv)
        queries = []

        for _, row in q_df.iterrows():
            cat = row['Variable']
            path = row['Path']

            if 'CheXpert-v1.0-small/' in path:
                path = path.replace('CheXpert-v1.0-small/', '')

            full_path = os.path.join(img_base_dir, path)

            queries.append({
                'image_path': full_path,
                'label': cat
            })

        cand_df = pd.read_csv(cand_csv)
        candidates = []

        cats = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                'Fracture', 'Support Devices']

        for _, row in cand_df.iterrows():
            path = row['Path']

            if 'CheXpert-v1.0-small/' in path:
                path = path.replace('CheXpert-v1.0-small/', '')

            full_path = os.path.join(img_base_dir, path)

            lbl = None
            for cat in cats:
                if cat in row and row[cat] == 1.0:
                    lbl = cat
                    break

            if lbl is not None:
                candidates.append({
                    'image_path': full_path,
                    'label': lbl
                })

        dataset = RetrievalDataset(
            query_data=queries,
            candidate_data=candidates,
            transform=val_tfm,
            is_text_query=False
        )
        return {'dataset': dataset}

    elif task_name == 'retrieval_txt':
        q_csv = os.path.join(eval_data_dir, 'text-retrieval/query.csv')
        cand_csv = os.path.join(eval_data_dir, 'image-retrieval/candidate.csv')  # Same candidates

        q_df = pd.read_csv(q_csv)
        queries = []

        for _, row in q_df.iterrows():
            cat = row['Variable']
            txt = row['Text']

            queries.append({
                'text': txt,
                'label': cat
            })

        cand_df = pd.read_csv(cand_csv)
        candidates = []

        cats = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                'Fracture', 'Support Devices']

        for _, row in cand_df.iterrows():
            path = row['Path']

            if 'CheXpert-v1.0-small/' in path:
                path = path.replace('CheXpert-v1.0-small/', '')

            full_path = os.path.join(img_base_dir, path)

            lbl = None
            for cat in cats:
                if cat in row and row[cat] == 1.0:
                    lbl = cat
                    break

            if lbl is not None:
                candidates.append({
                    'image_path': full_path,
                    'label': lbl
                })

        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        dataset = RetrievalDataset(
            query_data=queries,
            candidate_data=candidates,
            transform=val_tfm,
            is_text_query=True,
            tokenizer=tokenizer
        )
        return {'dataset': dataset}
    else:
        raise ValueError(f"Unknown task: {task_name}")


def build_dataloaders(dataset, batch_size=32, num_workers=4, shuffle=True):
    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        return result

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return loader