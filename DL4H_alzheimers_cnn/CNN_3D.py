"""
CNN_3D.py
Trains and evaluates SimpleCNN3D and ResNet2D+3DAdapter on 3D NIfTI
brain MRI volumes for 4-class Alzheimer's classification.
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
from torch.utils.data import TensorDataset, DataLoader

# ── reproducibility ──────────────────────────────────────────────────────────
_START_RUNTIME = time.time()

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

NUM_CLASSES = 4
CLASS_NAMES = ['none', 'verymild', 'mild', 'moderate']

# ── device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# ── output directory ─────────────────────────────────────────────────────────
RESULTS_DIR = 'results_3d'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'All outputs will be saved to: {RESULTS_DIR}/')


# ── data loading ─────────────────────────────────────────────────────────────
def load_from_pt(train_path, val_path, batch_size=2):
    """Load pre-saved .pt files and return DataLoaders for 3D volumes."""
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)

    train_ds = TensorDataset(train_data['imgs'], train_data['labels'])
    val_ds = TensorDataset(val_data['imgs'], val_data['labels'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f'Loaded train: {len(train_ds)} volumes, val: {len(val_ds)} volumes')
    return train_loader, val_loader


train_loader, val_loader = load_from_pt(
    train_path='nifti_train.pt',
    val_path='nifti_val.pt'
)


# ── verify data ──────────────────────────────────────────────────────────────
def print_counts(dataset, name):
    labels = dataset.tensors[1]
    classes, counts = torch.unique(labels, return_counts=True)
    print(f'--- {name} Dataset ---')
    for cls, count in zip(classes, counts):
        cls_name = CLASS_NAMES[cls.item()] if cls.item() < len(CLASS_NAMES) else f'class_{cls.item()}'
        print(f'  {cls_name} (class {cls.item()}): {count.item()} volumes')


print_counts(train_loader.dataset, 'Train')
print_counts(val_loader.dataset, 'Validation')

imgs, labels = next(iter(train_loader))
print(f'\nBatch shape: {imgs.shape}')  # expect (2, 1, 91, 109, 91)
print(f'Labels: {labels}')
print(f'Dtype: {imgs.dtype}')


# ── models ───────────────────────────────────────────────────────────────────
class ResNet2DWith3DAdapter(nn.Module):
    """
    Wraps a pretrained 2D ResNet18 with a Conv3d adapter layer in front.

    The adapter collapses the depth dimension of 3D MRI volumes into 3 RGB-like channels:
        (batch, 1, 91, 109, 91) -> Conv3d(1->3, k=(91,1,1)) -> (batch, 3, 1, 109, 91)
        -> squeeze -> (batch, 3, 109, 91) -> pretrained 2D ResNet18 -> (batch, 4)

    Trainable: adapter Conv3d + final FC layer
    Frozen: all pretrained ResNet18 conv/bn layers
    """
    def __init__(self, weight_path, num_classes=4, depth=91):
        super().__init__()
        from torchvision import models

        # 3D-to-2D adapter: collapse depth into 3 channels
        self.adapter = nn.Conv3d(1, 3, kernel_size=(depth, 1, 1))

        # Load pretrained 2D ResNet18
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # match saved weights
        self.resnet.load_state_dict(
            torch.load(weight_path, map_location='cpu', weights_only=False)
        )
        # Replace fc for new number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Freeze all resnet layers except fc
        for name, param in self.resnet.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def forward(self, x):
        # x: (batch, 1, D, H, W)
        x = self.adapter(x)          # (batch, 3, 1, H, W)
        x = x.squeeze(2)             # (batch, 3, H, W)
        x = self.resnet(x)           # (batch, num_classes)
        return x


class SimpleCNN3D(nn.Module):
    def __init__(self):
        super(SimpleCNN3D, self).__init__()
        # Input: (1, 91, 109, 91)
        self.conv1 = nn.Conv3d(1, 8, 3)    # -> (8, 89, 107, 89)
        self.conv2 = nn.Conv3d(8, 6, 2)    # -> (6, 88, 106, 88)
        self.conv3 = nn.Conv3d(6, 3, 3)    # -> (3, 86, 104, 86)
        self.pool3 = nn.MaxPool3d(4, 4)    # -> (3, 21, 26, 21)
        # 3 * 21 * 26 * 21 = 34,398
        self.fc1 = nn.Linear(34398, 10)
        self.fc2 = nn.Linear(10, NUM_CLASSES)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── instantiate models ──────────────────────────────────────────────────────
WEIGHT_PATH = 'resnet18_weights_9.pth'
resnet3d_model = ResNet2DWith3DAdapter(weight_path=WEIGHT_PATH, num_classes=NUM_CLASSES, depth=91)

trainable = sum(p.numel() for p in resnet3d_model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in resnet3d_model.parameters() if not p.requires_grad)
total = trainable + frozen
print(f'\nResNet2D+3DAdapter total params: {total:,}')
print(f'  Trainable (adapter + fc): {trainable:,}')
print(f'  Frozen (pretrained body): {frozen:,}')

test_input = torch.randn(1, 1, 91, 109, 91)
test_output = resnet3d_model(test_input)
print(f'Test forward pass: input {test_input.shape} -> output {test_output.shape}')

simple3d_model = SimpleCNN3D()
print(f'\nSimpleCNN3D total params: {sum(p.numel() for p in simple3d_model.parameters()):,}')

test_output = simple3d_model(test_input)
print(f'Test forward pass: input {test_input.shape} -> output {test_output.shape}')


# ── parameter summary ────────────────────────────────────────────────────────
def print_param_summary(model, name):
    print(f'\n=== {name} Parameter Summary ===')
    print(f'{"Layer Name":<60} | {"Type":<10} | {"Parameters":<12}')
    print('-' * 85)
    for pname, param in model.named_parameters():
        status = 'Trainable' if param.requires_grad else 'Frozen'
        print(f'{pname:<60} | {status:<10} | {param.numel():>12,}')
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print('-' * 85)
    print(f'Total Frozen:    {f:,}')
    print(f'Total Trainable: {t:,}')
    size_gb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e9
    print(f'Model size: {size_gb:.6f} GB')


print_param_summary(resnet3d_model, 'ResNet3D')
print_param_summary(simple3d_model, 'SimpleCNN3D')

# ── loss and optimisers ──────────────────────────────────────────────────────
# Class weights: none(268), verymild(56), mild(22), moderate(2)
weights = torch.tensor([1.0, 5.0, 12.0, 134.0]).to(device)

resnet3d_criterion = nn.CrossEntropyLoss(weight=weights)
resnet3d_optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, resnet3d_model.parameters()),
    lr=0.001
)

simple3d_criterion = nn.CrossEntropyLoss(weight=weights)
simple3d_optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, simple3d_model.parameters()),
    lr=0.001
)

print(f'\nClass weights: {weights}')


# ── checkpoint helpers ───────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


# ── evaluation ───────────────────────────────────────────────────────────────
def eval_model(model, dataloader, device):
    model.eval()
    Y_pred, Y_true, Y_scores = [], [], []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            Y_float = model(data)
            probs = torch.nn.functional.softmax(Y_float, dim=1)
            Y_scores.append(probs.detach().cpu().numpy())
            Y_pred.append(torch.nn.functional.one_hot(
                torch.argmax(Y_float, dim=1), num_classes=NUM_CLASSES
            ).detach().cpu().numpy())
            Y_true.append(torch.nn.functional.one_hot(
                target, num_classes=NUM_CLASSES
            ).detach().cpu().numpy())
    return (np.concatenate(Y_pred, axis=0),
            np.concatenate(Y_true, axis=0),
            np.concatenate(Y_scores, axis=0))


# ── training loop ────────────────────────────────────────────────────────────
def train_model(model, train_dataloader, val_dataloader, n_epoch, optimizer,
                criterion, checkpoint_path, file_name_csv, device):
    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f'Found checkpoint at {checkpoint_path}. Loading...')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f'Resuming from epoch {start_epoch}')
    else:
        print('No checkpoint found. Starting from scratch.')

    loss_train = []
    loss_val = []
    f1_val = []

    for epoch in range(n_epoch):
        curr_epoch_loss = []
        model.train()

        for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            target = target.to(device)
            predict = model(data)
            loss = criterion(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())

        avg_loss = np.mean(curr_epoch_loss)
        print(f'Epoch {epoch}: train_loss={avg_loss:.6f}')
        save_checkpoint(epoch, model, optimizer, avg_loss, checkpoint_path)

        loss_train.append(avg_loss)

        # Evaluate
        y_pred, y_true, y_scores = eval_model(model, val_dataloader, device)
        y_pred_idx = np.argmax(y_pred, axis=1)
        y_true_idx = np.argmax(y_true, axis=1)

        val_acc = accuracy_score(y_true_idx, y_pred_idx)
        f1 = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)

        loss_val.append(val_acc)
        f1_val.append(f1)
        print(f'Epoch {epoch}: val_acc={val_acc:.4f}, val_f1_macro={f1:.4f}')

    # Save metrics to CSV
    df = pd.DataFrame({
        'train_loss': loss_train,
        'val_accuracy': loss_val,
        'val_f1_macro': f1_val
    })
    df.to_csv(file_name_csv, index=False)
    print(f"Metrics saved to '{file_name_csv}'")

    return model


# ── move models to device ────────────────────────────────────────────────────
resnet3d_model = resnet3d_model.to(device)
simple3d_model = simple3d_model.to(device)
print('\nModels moved to', device)

# ── train ────────────────────────────────────────────────────────────────────
n_epoch = 20

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

resnet3d_model = train_model(
    resnet3d_model, train_loader, val_loader,
    optimizer=resnet3d_optimizer, n_epoch=n_epoch,
    criterion=resnet3d_criterion,
    checkpoint_path=os.path.join(RESULTS_DIR, 'resnet3d_checkpoint.pt'),
    file_name_csv=os.path.join(RESULTS_DIR, 'resnet3d.csv'),
    device=device
)

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

simple3d_model = train_model(
    simple3d_model, train_loader, val_loader,
    optimizer=simple3d_optimizer, n_epoch=n_epoch,
    criterion=simple3d_criterion,
    checkpoint_path=os.path.join(RESULTS_DIR, 'simple3d_checkpoint.pt'),
    file_name_csv=os.path.join(RESULTS_DIR, 'simple3d.csv'),
    device=device
)

# ── load checkpoints (if re-running evaluation only) ────────────────────────
# cp_path = os.path.join(RESULTS_DIR, 'resnet3d_checkpoint.pt')
# if os.path.exists(cp_path):
#     print(f'Loading {cp_path}...')
#     checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
#     resnet3d_model.load_state_dict(checkpoint['model_state_dict'])
#     resnet3d_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     print(f'Loaded epoch {checkpoint["epoch"]}')
#
# cp_path = os.path.join(RESULTS_DIR, 'simple3d_checkpoint.pt')
# if os.path.exists(cp_path):
#     print(f'Loading {cp_path}...')
#     checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
#     simple3d_model.load_state_dict(checkpoint['model_state_dict'])
#     simple3d_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     print(f'Loaded epoch {checkpoint["epoch"]}')


# ── model evaluation ─────────────────────────────────────────────────────────
def model_sum(model, model_str, val_loader, device):
    """Evaluate model and save confusion matrix, ROC curve, PR curve for multi-class."""
    y_pred, y_true, y_scores = eval_model(model, val_loader, device)

    y_pred_idx = np.argmax(y_pred, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)

    print(f'Accuracy:          {accuracy_score(y_true_idx, y_pred_idx):.4f}')
    print(f'Precision (macro): {precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0):.4f}')
    print(f'Recall (macro):    {recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0):.4f}')
    print(f'F1 (macro):        {f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0):.4f}')

    # Per-class metrics
    print('\nPer-class F1:')
    f1s = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        f1_val = f1s[i] if i < len(f1s) else 0.0
        print(f'  {name}: {f1_val:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES[:cm.shape[0]], columns=CLASS_NAMES[:cm.shape[1]])
    cm_path = os.path.join(RESULTS_DIR, f'{model_str}_confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f'\nConfusion Matrix:\n{cm_df}')
    print(f'Saved to {cm_path}')

    # Multi-class ROC (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'darkorange', 'green', 'red']
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if i >= y_true.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        auc_score = roc_auc_score(y_true[:, i], y_scores[:, i]) if y_true[:, i].sum() > 0 else 0.0
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_str} ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f'{model_str}_roc_curve.pdf')
    plt.savefig(roc_path)
    plt.close()
    print(f'ROC curve saved to {roc_path}')

    # Multi-class PR Curve (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if i >= y_true.shape[1]:
            continue
        precisions, recalls, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc_score = auc(recalls, precisions)
        ax.plot(recalls, precisions, color=color, lw=2, label=f'{name} (PR-AUC={pr_auc_score:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{model_str} PR Curves (One-vs-Rest)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(RESULTS_DIR, f'{model_str}_pr_curve.pdf')
    plt.savefig(pr_path)
    plt.close()
    print(f'PR curve saved to {pr_path}')

    return y_pred, y_true, y_scores


print('\n=== ResNet3D Evaluation ===')
resnet3d_pred, resnet3d_true, resnet3d_scores = model_sum(
    model=resnet3d_model, model_str='resnet3d', val_loader=val_loader, device=device
)

print('\n=== SimpleCNN3D Evaluation ===')
simple3d_pred, simple3d_true, simple3d_scores = model_sum(
    model=simple3d_model, model_str='simple3d', val_loader=val_loader, device=device
)


# ── ROC comparison plot ──────────────────────────────────────────────────────
def plot_roc_comparison(model1, name1, model2, name2, dataloader, device):
    """Plot macro-average ROC comparison between two models."""
    results = []
    for model, name in [(model1, name1), (model2, name2)]:
        y_pred, y_true, y_scores = eval_model(model, dataloader, device)
        fpr_all, tpr_all = [], []
        for i in range(NUM_CLASSES):
            if y_true[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
                fpr_all.append(fpr)
                tpr_all.append(tpr)
        macro_auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovr')
        results.append((fpr_all, tpr_all, macro_auc, name))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'darkorange']
    for (fpr_all, tpr_all, macro_auc, name), color in zip(results, colors):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(fpr_all, tpr_all):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(fpr_all)
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f'{name} (macro AUC={macro_auc:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison (Macro-Average)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'roc_curve_comparison.pdf')
    plt.savefig(path)
    plt.close()
    print(f'Saved to {path}')


plot_roc_comparison(resnet3d_model, 'ResNet3D', simple3d_model, 'SimpleCNN3D',
                    val_loader, device)


# ── PR comparison plot ───────────────────────────────────────────────────────
def plot_pr_comparison(model1, name1, model2, name2, dataloader, device):
    """Plot macro-average PR comparison between two models."""
    results = []
    for model, name in [(model1, name1), (model2, name2)]:
        y_pred, y_true, y_scores = eval_model(model, dataloader, device)
        pr_aucs = []
        recalls_all, precisions_all = [], []
        for i in range(NUM_CLASSES):
            if y_true[:, i].sum() > 0:
                precs, recs, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
                pr_aucs.append(auc(recs, precs))
                recalls_all.append(recs)
                precisions_all.append(precs)
        macro_pr_auc = np.mean(pr_aucs)
        results.append((recalls_all, precisions_all, macro_pr_auc, name))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'darkorange']
    for (recalls_all, precisions_all, macro_pr_auc, name), color in zip(results, colors):
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(mean_recall)
        for recs, precs in zip(recalls_all, precisions_all):
            mean_precision += np.interp(mean_recall, recs[::-1], precs[::-1])
        mean_precision /= len(recalls_all)
        ax.plot(mean_recall, mean_precision, color=color, lw=2,
                label=f'{name} (macro PR-AUC={macro_pr_auc:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve Comparison (Macro-Average)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'pr_comparison.pdf')
    plt.savefig(path)
    plt.close()
    print(f'Saved to {path}')


plot_pr_comparison(resnet3d_model, 'ResNet3D', simple3d_model, 'SimpleCNN3D',
                   val_loader, device)

# ── runtime ──────────────────────────────────────────────────────────────────
elapsed = time.time() - _START_RUNTIME
print(f'\nTotal runtime: {elapsed/60:.1f} minutes')
