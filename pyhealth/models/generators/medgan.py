"""MedGAN: Medical Generative Adversarial Network for synthetic EHR generation.

This is a port of the reference implementation
(https://github.com/mp2893/medgan and the PyTorch reimplementation under
``reference/cor-gan/Generative/medGAN/MIMIC/pytorch/MLP/medGAN.py``) wrapped
as a PyHealth ``BaseModel`` so it consumes the standard
``dataset -> SampleDataset -> model`` pipeline.

MedGAN treats each patient as a flat bag-of-codes (no visit structure), so it
expects an input feature named ``visits`` backed by a ``MultiHotProcessor``.
The training procedure has two phases (mirroring the reference):

* a **linear autoencoder** is pre-trained with binary cross-entropy
  reconstruction loss, and
* an **adversarial training** phase where the generator emits latent codes,
  the autoencoder's decoder projects them back to a multi-hot patient vector,
  and a discriminator with optional minibatch averaging tries to distinguish
  real from synthetic.

The ``MedGANAutoencoder``, ``MedGANGenerator`` and ``MedGANDiscriminator``
modules below mirror the reference. The public ``MedGAN`` class follows the
same API style as :class:`pyhealth.models.generators.HALO`
(``train_model`` / ``generate`` / ``save_model`` / ``load_model``).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from pyhealth.models import BaseModel


# ----------------------------------------------------------------------------
# Building blocks (ported from reference medgan.py / PyTorch reimplementation)
# ----------------------------------------------------------------------------
class _MultiHotDataset(Dataset):
    """Tiny ``torch.utils.data.Dataset`` over a multi-hot numpy matrix."""

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class MedGANAutoencoder(nn.Module):
    """Linear autoencoder for MedGAN pretraining.

    Mirrors the reference single-layer encoder/decoder
    (``Linear -> Tanh`` and ``Linear -> Sigmoid``).

    Args:
        input_dim: Vocabulary size (number of distinct codes).
        embedding_dim: Latent dimensionality. Default: 128.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class MedGANGenerator(nn.Module):
    """Two-layer MLP generator with residual connections (per reference)."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.act1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.act2 = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.linear1(x))) + residual

        residual = out
        out = self.act2(self.bn2(self.linear2(out))) + residual
        return out


class MedGANDiscriminator(nn.Module):
    """MLP discriminator with optional minibatch averaging (per reference)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        minibatch_averaging: bool = True,
    ):
        super().__init__()
        self.minibatch_averaging = minibatch_averaging
        model_input_dim = input_dim * 2 if minibatch_averaging else input_dim

        self.model = nn.Sequential(
            nn.Linear(model_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.minibatch_averaging:
            # Average over the batch and concatenate to each sample, exactly
            # as in the reference (medGAN.py).
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)
        return self.model(x)


def _weights_init(m):
    """Xavier-uniform for Linear, N(1, 0.02) gamma / 0 beta for BatchNorm."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0)


def _autoencoder_loss(x_output, y_target):
    """Sparse-friendly BCE: sum over features, mean over batch.

    Equivalent to ``BCELoss(reduction='sum') / batch_size`` and matches the
    reference; ``BCELoss(reduction='mean')`` would also mean over features
    which dilutes the signal for sparse code vectors.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (
        1.0 - y_target
    ) * torch.log(1.0 - x_output + epsilon)
    return torch.mean(-torch.sum(term, dim=1), dim=0)


# ----------------------------------------------------------------------------
# PyHealth BaseModel wrapper
# ----------------------------------------------------------------------------
class MedGAN(BaseModel):
    """MedGAN synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Generates synthetic binary EHR records via the two-phase procedure from
    Choi et al. (MLHC 2017): pretrain a linear autoencoder, then run BCE-GAN
    adversarial training where the generator maps noise to the autoencoder's
    latent space and the decoder projects back to a multi-hot patient vector.

    Generation is **unconditional**: each synthetic patient is a flat bag of
    codes (no visit structure), matching the ``multi_hot`` input schema.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": "multi_hot"}`` and whose ``output_schema`` is empty.
        latent_dim: Generator noise dimensionality. Default: 128. The
            generator's residual connection requires ``latent_dim ==
            hidden_dim``; if they differ, ``latent_dim`` is silently aligned to
            ``hidden_dim``.
        hidden_dim: Generator hidden width (also the autoencoder embedding
            dimension). Default: 128.
        discriminator_hidden_dim: Discriminator hidden width. Default: 256.
        minibatch_averaging: Concatenate per-batch mean to each discriminator
            input. Default: True.
        batch_size: Training batch size. Default: 512.
        ae_epochs: Autoencoder pre-training epochs. Default: 100.
        gan_epochs: Adversarial training epochs. Default: 200.
        ae_lr: Autoencoder learning rate. Default: 1e-3.
        gan_lr: GAN learning rate. Default: 1e-3.
        save_dir: Checkpoint directory used by ``train_model``.
            Default: ``"./save/medgan/"``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": ["A", "B", "C"]},
        ...     {"patient_id": "p2", "visits": ["A", "C", "D"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "multi_hot"},
        ...     output_schema={},
        ... )
        >>> model = MedGAN(dataset, latent_dim=16, hidden_dim=16, batch_size=2)
        >>> isinstance(model, MedGAN)
        True
    """

    def __init__(
        self,
        dataset,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        minibatch_averaging: bool = True,
        batch_size: int = 512,
        ae_epochs: int = 100,
        gan_epochs: int = 200,
        ae_lr: float = 1e-3,
        gan_lr: float = 1e-3,
        save_dir: str = "./save/medgan/",
    ) -> None:
        super().__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "MedGAN expects an input feature named 'visits' backed by a "
                "MultiHotProcessor."
            )

        # The generator's residual connection (``out + residual`` with
        # ``residual`` being the noise input) requires latent_dim == hidden_dim.
        # Align silently if the user mismatched, mirroring CorGAN.
        if latent_dim != hidden_dim:
            latent_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self._batch_size = batch_size
        self._ae_epochs = ae_epochs
        self._gan_epochs = gan_epochs
        self._ae_lr = ae_lr
        self._gan_lr = gan_lr
        self.save_dir = save_dir

        # Code vocab from the MultiHotProcessor's label_vocab.
        self.visits_processor = dataset.input_processors["visits"]
        self.input_dim = self.visits_processor.size()
        self._idx_to_code: List[Optional[str]] = [None] * self.input_dim
        for code, idx in self.visits_processor.label_vocab.items():
            self._idx_to_code[idx] = code

        self.autoencoder = MedGANAutoencoder(
            input_dim=self.input_dim,
            embedding_dim=hidden_dim,
        )
        self.generator = MedGANGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.discriminator = MedGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging,
        )

        self.autoencoder.apply(_weights_init)
        self.generator.apply(_weights_init)
        self.discriminator.apply(_weights_init)

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """MedGAN does not have a single supervised forward pass.

        Use :meth:`train_model` for training and :meth:`generate` for
        synthesis. ``forward`` is implemented only to satisfy the
        ``BaseModel`` abstract contract.
        """
        raise NotImplementedError(
            "MedGAN is a GAN: use train_model() and generate() instead of "
            "forward()."
        )

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        """Resolve a user-supplied device, defaulting to CUDA when available."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_dataloader(self, dataset) -> DataLoader:
        """Stack the multi-hot tensors of ``dataset`` into a DataLoader.

        The fitted ``MultiHotProcessor`` has already converted each patient's
        ``visits`` field into a ``(input_dim,)`` float32 tensor, so we can
        simply stack and wrap.
        """
        tensors = [dataset[i]["visits"] for i in range(len(dataset))]
        matrix = torch.stack(tensors).numpy()
        wrapped = _MultiHotDataset(matrix)
        sampler = RandomSampler(wrapped, replacement=True)
        return DataLoader(
            wrapped,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
        )

    def train_model(self, train_dataset, val_dataset=None, device=None) -> None:
        """Train MedGAN with a custom two-phase loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Phase 1 pre-trains the autoencoder with
        sparse-friendly BCE reconstruction loss; phase 2 runs standard
        BCE-GAN adversarial training where the generator+decoder are
        optimised against a binary discriminator.

        Args:
            train_dataset: ``SampleDataset`` for training.
            val_dataset: Unused; accepted for API symmetry with other PyHealth
                trainers.
            device: Device to train on (``"cuda"``, ``"cpu"``, etc.). If
                ``None``, uses CUDA when available.
        """
        device = self._resolve_device(device)
        self.to(device)
        print(f"Training MedGAN on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        dataloader = self._build_dataloader(train_dataset)

        # ---- Phase 1: Autoencoder pretraining ----
        optimizer_ae = torch.optim.Adam(
            self.autoencoder.parameters(), lr=self._ae_lr
        )
        for epoch in tqdm(range(self._ae_epochs), desc="AE pretrain"):
            self.autoencoder.train()
            total_loss, n_batches = 0.0, 0
            for batch in dataloader:
                real = batch.to(self.device)
                recon = self.autoencoder(real)
                loss = _autoencoder_loss(recon, real)

                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()

                total_loss += loss.item()
                n_batches += 1

        # ---- Phase 2: Adversarial training ----
        # Generator + the autoencoder's decoder are trained jointly, matching
        # the reference (the decoder is what makes synthetic samples valid).
        optimizer_g = torch.optim.Adam(
            list(self.generator.parameters())
            + list(self.autoencoder.decoder.parameters()),
            lr=self._gan_lr,
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self._gan_lr
        )

        best_d_loss = float("inf")
        for epoch in tqdm(range(self._gan_epochs), desc="GAN train"):
            self.generator.train()
            self.discriminator.train()
            self.autoencoder.eval()
            self.autoencoder.decoder.train()

            epoch_d_loss, epoch_g_loss, n_batches = 0.0, 0.0, 0
            for batch in dataloader:
                real = batch.to(self.device)
                bs = real.size(0)

                # --- Train Discriminator ---
                optimizer_d.zero_grad()
                noise = torch.randn(bs, self.latent_dim, device=self.device)
                fake = self.autoencoder.decode(self.generator(noise))

                real_pred = self.discriminator(real)
                fake_pred = self.discriminator(fake.detach())

                d_loss = F.binary_cross_entropy(
                    real_pred, torch.ones_like(real_pred)
                ) + F.binary_cross_entropy(
                    fake_pred, torch.zeros_like(fake_pred)
                )
                d_loss.backward()
                optimizer_d.step()

                # --- Train Generator (+ decoder) ---
                optimizer_g.zero_grad()
                fake_pred = self.discriminator(fake)
                g_loss = F.binary_cross_entropy(
                    fake_pred, torch.ones_like(fake_pred)
                )
                g_loss.backward()
                optimizer_g.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                n_batches += 1

            avg_d = epoch_d_loss / max(n_batches, 1)
            if avg_d < best_d_loss:
                best_d_loss = avg_d
                self.save_model(os.path.join(self.save_dir, "best.pt"))

        self.save_model(os.path.join(self.save_dir, "final.pt"))

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def generate(
        self,
        num_samples: int,
        random_sampling: bool = False,
        device=None,
    ) -> List[Dict]:
        """Generate synthetic patient records.

        Each synthetic patient is decoded from a generated multi-hot vector
        by thresholding (or, optionally, Bernoulli sampling) at 0.5 and
        mapping the indices back to code strings.

        Args:
            num_samples: Number of synthetic patients to generate.
            random_sampling: If True, Bernoulli-sample the decoder output;
                otherwise threshold at 0.5 (the reference's behaviour).
                Default: False.
            device: Device to generate on. If ``None``, uses CUDA when
                available.

        Returns:
            List of dicts
            ``{"patient_id": "synthetic_i", "visits": [[code, ...]]}``.
            ``visits`` is a list containing a **single** visit (matching
            HALO's nested-list output structure). MedGAN is a bag-of-codes
            model -- following the reference ``process_mimic.py``, each
            patient is represented by the union of codes across all of
            their historical visits -- so the single inner list is that
            aggregate bag. The inner list may be empty if the generator
            produced an all-zero vector.
        """
        device = self._resolve_device(device)
        self.to(device)

        self.generator.eval()
        self.autoencoder.eval()

        bs = min(self._batch_size, max(num_samples, 1))
        rows = np.zeros((num_samples, self.input_dim), dtype=np.float32)
        pbar = tqdm(total=num_samples, desc="Generating patients")
        with torch.no_grad():
            i = 0
            while i < num_samples:
                cur = min(bs, num_samples - i)
                z = torch.randn(cur, self.latent_dim, device=self.device)
                probs = self.autoencoder.decode(self.generator(z))
                if random_sampling:
                    sample = torch.bernoulli(probs)
                else:
                    sample = (probs >= 0.5).float()
                rows[i : i + cur] = sample.cpu().numpy()
                i += cur
                pbar.update(cur)
        pbar.close()

        results: List[Dict] = []
        for i in range(num_samples):
            codes = [
                self._idx_to_code[idx]
                for idx in np.nonzero(rows[i])[0]
                if self._idx_to_code[idx] not in (None, "<pad>", "<unk>")
            ]
            # Wrap in a single-visit list to mirror HALO's nested output.
            # MedGAN models the patient as one aggregate bag of codes.
            results.append({"patient_id": f"synthetic_{i}", "visits": [codes]})
        return results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        """Save weights and the code vocabulary needed for decoding."""
        torch.save(
            {
                "autoencoder": self.autoencoder.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "idx_to_code": self._idx_to_code,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load weights and the code vocabulary from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        if "idx_to_code" in ckpt:
            self._idx_to_code = ckpt["idx_to_code"]
