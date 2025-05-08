#!/usr/bin/env python
# train.py
import argparse
import sklearn.metrics
import os
import sys
sys.path.append(os.path.dirname(__file__))  # Add current directory to path

from models import CXRClassifier, CXRAdvClassifier
from cxrdataset import NIHDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _find_index(ds, desired_label):
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            return ilabel
    raise ValueError(f"Label {desired_label} not found.")

def _train_standard(datasetclass, checkpoint_path, logpath):
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')

    classifier = CXRClassifier()
    classifier.train(trainds,
                     valds,
                     max_epochs=10,
                     batch_size=8,
                     lr=0.01, 
                     weight_decay=1e-4,
                     logpath=logpath,
                     checkpoint_path=checkpoint_path,
                     verbose=True)

    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    pneumonia_index = _find_index(testds, 'pneumonia')
    auroc = sklearn.metrics.roc_auc_score(true[:, pneumonia_index], probs[:, pneumonia_index])
    print(f"AUROC for pneumonia: {auroc:.4f}")

def _train_adversarial(datasetclass, checkpoint_path, logpath):
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')

    classifier = CXRAdvClassifier()
    classifier.train(trainds,
                     valds,
                     max_epochs=10,
                     batch_size=8,
                     lr=0.01,
                     weight_decay=1e-4,
                     logpath=logpath,
                     checkpoint_path=checkpoint_path,
                     verbose=True)

    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    pneumonia_index = _find_index(testds, 'pneumonia')
    auroc = sklearn.metrics.roc_auc_score(true[:, pneumonia_index], probs[:, pneumonia_index])
    print(f"AUROC for pneumonia: {auroc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['NIH'], help='Dataset to use')
    parser.add_argument('training', choices=['Standard', 'Adversarial'], help='Training strategy')
    args = parser.parse_args()

    if args.dataset == 'NIH' and args.training == 'Standard':
        _train_standard(NIHDataset, 'nih_standard_model.pkl', 'nih_standard.log')
    elif args.dataset == 'NIH' and args.training == 'Adversarial':
        _train_adversarial(NIHDataset, 'nih_adversarial_model.pkl', 'nih_adversarial.log')
    else:
        raise ValueError("Unsupported config.")

if __name__ == '__main__':
    main()
