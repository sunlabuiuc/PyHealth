"""
Adaptive Transfer Learning Model for Physical Activity Monitoring.

Implements the three-stage framework from Zhang et al. CHIL 2024:
  1. Inter-domain Pairwise Distance (IPD) computation
  2. IPD-guided sequential pre-training on source domains
  3. Fine-tuning on target domain with k-fold cross-validation

Reference:
    Zhang et al. "Daily Physical Activity Monitoring: Adaptive Learning
    from Multi-source Motion Sensor Data." CHIL 2024.
    https://github.com/Oceanjinghai/HealthTimeSerial
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple


# ── Backbone Classifiers ───────────────────────────────────────────────────────

class LSTMBackbone(nn.Module):
    """LSTM-based classifier for time series classification.

    Args:
        input_size (int): Number of input channels. Default 9.
        hidden_size (int): LSTM hidden dimension. Default 128.
        num_layers (int): Number of LSTM layers. Default 2.
        num_classes (int): Number of output classes. Default 19.
        dropout (float): Dropout rate. Default 0.3.

    Examples:
        >>> model = LSTMBackbone()
        >>> x = torch.randn(32, 9, 125)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([32, 19])
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 19,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: Shape (batch, num_classes) logits.
        """
        # LSTM expects (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1, :])  # last timestep
        return out


class ResNetBackbone(nn.Module):
    """Residual CNN classifier for time series classification.

    Args:
        input_size (int): Number of input channels. Default 9.
        num_classes (int): Number of output classes. Default 19.

    Examples:
        >>> model = ResNetBackbone()
        >>> x = torch.randn(32, 9, 125)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([32, 19])
    """

    def __init__(
        self,
        input_size: int = 9,
        num_classes: int = 19,
    ) -> None:
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=8, padding=4),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
            )

        self.block1 = _block(input_size, 64)
        self.skip1  = nn.Conv1d(input_size, 64, kernel_size=1)
        self.block2 = _block(64, 128)
        self.skip2  = nn.Conv1d(64, 128, kernel_size=1)
        self.block3 = _block(128, 128)
        self.skip3  = nn.Identity()
        self.relu   = nn.ReLU()
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: Shape (batch, num_classes) logits.
        """
        x = self.relu(self.block1(x) + self.skip1(x))
        x = self.relu(self.block2(x) + self.skip2(x))
        x = self.relu(self.block3(x) + self.skip3(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


BACKBONES = {"lstm": LSTMBackbone, "resnet": ResNetBackbone}


# ── IPD Computation ────────────────────────────────────────────────────────────

def compute_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two 1D sequences.

    Args:
        s1 (np.ndarray): Shape (T,) first sequence.
        s2 (np.ndarray): Shape (T,) second sequence.

    Returns:
        float: DTW distance.
    """
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(dtw[n, m])


def compute_ipd(
    source_X: np.ndarray,
    target_X: np.ndarray,
    n_samples: int = 200,
    distance: str = "euclidean",
) -> float:
    """Compute Inter-domain Pairwise Distance (IPD) between source and target.

    Implements Algorithm 1 from the paper:
    1. Compute per-channel pairwise distances for each matched pair
    2. Fit Gaussian KDE to the difference vectors
    3. Sample from KDE and return matrix norm as IPD estimate

    Args:
        source_X (np.ndarray): Shape (N, 9, 125) source domain samples.
        target_X (np.ndarray): Shape (N, 9, 125) target domain samples.
        n_samples (int): Number of Monte Carlo samples from KDE. Default 200.
        distance (str): Distance metric. "euclidean" or "dtw". Default "euclidean".

    Returns:
        float: Estimated IPD score. Higher = more different domains.
    """
    N = min(len(source_X), len(target_X), 100)  # cap for speed
    K = source_X.shape[1]  # number of channels (9)

    # Step 1: Compute empirical difference vectors (Algorithm 1)
    diff_vectors = []
    for n in range(N):
        s_vec = []
        for k in range(K):
            s1 = source_X[n, k, :]
            s2 = target_X[n, k, :]
            if distance == "dtw":
                d = compute_dtw_distance(s1, s2)
            else:
                d = float(np.linalg.norm(s1 - s2))
            s_vec.append(d)
        diff_vectors.append(s_vec)

    M = np.array(diff_vectors, dtype=np.float32)  # shape (N, K)

    # Step 2: Fit Gaussian KDE (smooth bootstrap)
    mean = M.mean(axis=0)         # (K,)
    cov  = np.cov(M.T) + np.eye(K) * 1e-6  # (K, K) regularized

    # Step 3: Sample from KDE and compute matrix norm
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    samples = np.abs(samples)     # distances are non-negative
    ipd = float(np.linalg.norm(samples) / n_samples)
    return ipd


# ── Main Model ─────────────────────────────────────────────────────────────────

class AdaptiveTransferModel(nn.Module):
    """Adaptive Transfer Learning model for physical activity classification.

    Implements the full three-stage pipeline from Zhang et al. CHIL 2024:

    Stage 1 - IPD Computation:
        Computes Inter-domain Pairwise Distance between each source domain
        and the target domain to quantify domain similarity.

    Stage 2 - Source Domain Pre-training:
        Trains the backbone classifier sequentially across source domains,
        sorted by descending IPD (most different first). The learning rate
        decays faster for dissimilar domains (higher IPD).

    Stage 3 - Target Domain Fine-tuning:
        Fine-tunes the pre-trained model on target domain data using
        mini-batch gradient descent with k-fold cross-validation.

    Args:
        backbone (str): Backbone classifier. "lstm" or "resnet". Default "lstm".
        num_classes (int): Number of activity classes. Default 19.
        input_channels (int): Number of sensor channels. Default 9.
        hidden_size (int): Hidden size for LSTM backbone. Default 128.
        lr (float): Initial learning rate. Default 5e-4.
        epochs_per_source (int): Training epochs per source domain. Default 50.
        epochs_target (int): Fine-tuning epochs on target domain. Default 100.
        k_folds (int): Number of folds for target domain CV. Default 10.
        max_degeneration (int): Early stopping patience. Default 5.
        lr_target (float): Baseline learning rate for target fine-tuning. Default 1e-3.
        distance (str): IPD distance metric. "euclidean" or "dtw". Default "euclidean".
        device (str): "cuda" or "cpu". Auto-detected if not specified.

    Examples:
        >>> model = AdaptiveTransferModel(backbone="lstm")
        >>> # source_datasets: dict of sensor_id -> PhysicalActivityDataset
        >>> # target_dataset: PhysicalActivityDataset
        >>> model.fit(source_datasets, target_dataset)
        >>> preds = model.predict(target_test_dataset)
    """

    def __init__(
        self,
        backbone: str = "lstm",
        num_classes: int = 19,
        input_channels: int = 9,
        hidden_size: int = 128,
        lr: float = 5e-4,
        epochs_per_source: int = 50,
        epochs_target: int = 100,
        k_folds: int = 10,
        max_degeneration: int = 5,
        lr_target: float = 1e-3,
        distance: str = "euclidean",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                "backbone must be one of {}, got '{}'.".format(
                    list(BACKBONES.keys()), backbone
                )
            )

        self.backbone_name = backbone
        self.num_classes   = num_classes
        self.lr            = lr
        self.epochs_per_source  = epochs_per_source
        self.epochs_target      = epochs_target
        self.k_folds            = k_folds
        self.max_degeneration   = max_degeneration
        self.lr_target          = lr_target
        self.distance           = distance
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build backbone
        if backbone == "lstm":
            self.classifier = LSTMBackbone(
                input_size=input_channels,
                hidden_size=hidden_size,
                num_classes=num_classes,
            )
        else:
            self.classifier = ResNetBackbone(
                input_size=input_channels,
                num_classes=num_classes,
            )

        self.classifier.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.ipd_scores: Dict[str, float] = {}

    # ── Stage 1: IPD Computation ───────────────────────────────────────────

    def compute_all_ipds(
        self,
        source_datasets: Dict,
        target_dataset,
    ) -> Dict[str, float]:
        """Compute IPD between each source domain and the target domain.

        Args:
            source_datasets (Dict): sensor_id -> PhysicalActivityDataset.
            target_dataset (PhysicalActivityDataset): Target domain dataset.

        Returns:
            Dict mapping sensor_id -> IPD score.
        """
        target_X = target_dataset.X.numpy()   # (N, 9, 125)

        ipd_scores = {}
        for sid, src_ds in source_datasets.items():
            src_X = src_ds.X.numpy()
            ipd = compute_ipd(
                src_X, target_X,
                n_samples=200,
                distance=self.distance,
            )
            ipd_scores[sid] = ipd
            print("  IPD({}) = {:.4f}".format(sid, ipd))

        self.ipd_scores = ipd_scores
        return ipd_scores

    # ── Stage 2: Source Domain Pre-training ───────────────────────────────

    def _get_alpha(self, sid: str) -> float:
        """Compute normalized IPD weight for one source domain.

        alpha_q = g_q / sum(g_1 ... g_Q)  (Equation 6 in paper)

        Args:
            sid (str): Source domain sensor ID.

        Returns:
            float: Normalized weight in [0, 1].
        """
        total = sum(self.ipd_scores.values())
        if total == 0:
            return 1.0 / len(self.ipd_scores)
        return self.ipd_scores[sid] / total

    def pretrain_on_sources(
        self,
        source_datasets: Dict,
    ) -> None:
        """Pre-train classifier sequentially on source domains.

        Domains are sorted by descending IPD (most different first,
        most similar to target last). Learning rate decays faster for
        more dissimilar domains, transferring more knowledge from
        similar domains.

        Args:
            source_datasets (Dict): sensor_id -> PhysicalActivityDataset.
        """
        # Sort by descending IPD (most different first)
        sorted_sources = sorted(
            source_datasets.items(),
            key=lambda x: self.ipd_scores.get(x[0], 0),
            reverse=True,
        )
        print("\nPre-training order (most different -> most similar):")
        for sid, _ in sorted_sources:
            print("  {} (IPD={:.4f})".format(sid, self.ipd_scores.get(sid, 0)))

        optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.lr
        )

        for sid, src_ds in sorted_sources:
            alpha = self._get_alpha(sid)
            loader = DataLoader(src_ds, batch_size=64, shuffle=True)
            current_lr = self.lr

            print("\nTraining on source {} | alpha={:.4f}".format(sid, alpha))
            for epoch in range(self.epochs_per_source):
                self.classifier.train()
                epoch_loss = 0.0
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    logits = self.classifier(X_batch)
                    loss = self.loss_fn(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Decay learning rate by alpha (Equation 5 in paper)
                current_lr = current_lr * (1 - alpha)
                for pg in optimizer.param_groups:
                    pg['lr'] = max(current_lr, 1e-6)

                if (epoch + 1) % 10 == 0:
                    print("  Epoch {}/{} | Loss={:.4f} | LR={:.6f}".format(
                        epoch + 1, self.epochs_per_source,
                        epoch_loss / len(loader), current_lr
                    ))

    # ── Stage 3: Target Domain Fine-tuning ────────────────────────────────

    def finetune_on_target(self, target_dataset) -> None:
        """Fine-tune the pre-trained model on the target domain.

        Uses mini-batch gradient descent with k-fold cross-validation.
        The learning rate adjusts based on validation performance.
        Stops early after max_degeneration consecutive performance drops.

        Args:
            target_dataset (PhysicalActivityDataset): Target domain data.
        """
        n = len(target_dataset)
        indices = torch.randperm(n).tolist()
        fold_size = n // self.k_folds
        folds = [
            indices[i * fold_size:(i + 1) * fold_size]
            for i in range(self.k_folds)
        ]

        optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.lr_target
        )

        current_lr = self.lr_target
        degeneration_count = 0
        prev_val_loss = float('inf')

        print("\nFine-tuning on target domain...")
        for epoch in range(self.epochs_target):
            # Randomly pick one fold as validation
            val_fold_idx = epoch % self.k_folds
            val_idx   = folds[val_fold_idx]
            train_idx = [
                i for j, fold in enumerate(folds)
                for i in fold if j != val_fold_idx
            ]

            train_loader = DataLoader(
                Subset(target_dataset, train_idx),
                batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                Subset(target_dataset, val_idx),
                batch_size=32, shuffle=False
            )

            # Training step
            self.classifier.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_fn(self.classifier(X_b), y_b)
                loss.backward()
                optimizer.step()

            # Validation step
            val_loss = self._eval_loss(val_loader)

            # Adjust LR based on val performance (Equation 9)
            current_lr = (1 - val_loss) * self.lr_target
            current_lr = max(current_lr, 1e-6)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # Early stopping
            if val_loss > prev_val_loss:
                degeneration_count += 1
            else:
                degeneration_count = 0
            prev_val_loss = val_loss

            if degeneration_count >= self.max_degeneration:
                print("  Early stopping at epoch {}".format(epoch + 1))
                break

            if (epoch + 1) % 20 == 0:
                print("  Epoch {}/{} | Val Loss={:.4f}".format(
                    epoch + 1, self.epochs_target, val_loss
                ))

    def _eval_loss(self, loader: DataLoader) -> float:
        """Compute average cross-entropy loss on a DataLoader.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            float: Average loss.
        """
        self.classifier.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                loss = self.loss_fn(self.classifier(X_b), y_b)
                total_loss += loss.item() * len(y_b)
                n += len(y_b)
        return total_loss / n if n > 0 else 0.0

    # ── Full Pipeline ──────────────────────────────────────────────────────

    def fit(
        self,
        source_datasets: Dict,
        target_train_dataset,
    ) -> None:
        """Run the full three-stage training pipeline.

        Stage 1: Compute IPD for all source domains.
        Stage 2: Pre-train on source domains guided by IPD.
        Stage 3: Fine-tune on target domain.

        Args:
            source_datasets (Dict): sensor_id -> PhysicalActivityDataset.
            target_train_dataset (PhysicalActivityDataset): Target training data.
        """
        print("=" * 50)
        print("Stage 1: Computing IPD scores...")
        print("=" * 50)
        self.compute_all_ipds(source_datasets, target_train_dataset)

        print("\n" + "=" * 50)
        print("Stage 2: Pre-training on source domains...")
        print("=" * 50)
        self.pretrain_on_sources(source_datasets)

        print("\n" + "=" * 50)
        print("Stage 3: Fine-tuning on target domain...")
        print("=" * 50)
        self.finetune_on_target(target_train_dataset)

        print("\nTraining complete!")

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, test_dataset) -> float:
        """Compute Ratio of Correct Classification (RCC) on test data.

        Args:
            test_dataset (PhysicalActivityDataset): Test dataset.

        Returns:
            float: RCC accuracy (0 to 1).
        """
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                preds = self.classifier(X_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += len(y_b)
        return correct / total if total > 0 else 0.0

    def predict(self, dataset) -> np.ndarray:
        """Generate predictions for a dataset.

        Args:
            dataset (PhysicalActivityDataset): Input dataset.

        Returns:
            np.ndarray: Predicted class indices, shape (N,).
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        self.classifier.eval()
        all_preds = []
        with torch.no_grad():
            for X_b, _ in loader:
                X_b = X_b.to(self.device)
                preds = self.classifier(X_b).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone classifier.

        Args:
            x (torch.Tensor): Shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: Shape (batch, num_classes) logits.
        """
        return self.classifier(x)
