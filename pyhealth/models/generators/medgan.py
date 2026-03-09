"""MedGAN: Medical Generative Adversarial Network for synthetic EHR generation.

Reference:
    Choi et al., "Generating Multi-label Discrete Patient Records using
    Generative Adversarial Networks", MLHC 2017.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pyhealth.models import BaseModel


class MedGANDataset(Dataset):
    """Dataset wrapper for MedGAN training from a numpy binary matrix."""

    def __init__(self, data):
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class MedGANAutoencoder(nn.Module):
    """Linear autoencoder for MedGAN pretraining.

    Args:
        input_dim (int): Dimensionality of the input (vocabulary size).
        hidden_dim (int): Dimensionality of the latent space. Default: 128.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class MedGANGenerator(nn.Module):
    """Generator with residual connections.

    Args:
        latent_dim (int): Dimensionality of the noise input. Default: 128.
        hidden_dim (int): Width of hidden layers. Default: 128.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.activation1(self.bn1(self.linear1(x)))
        out1 = out + residual

        residual = out1
        out = self.activation2(self.bn2(self.linear2(out1)))
        out2 = out + residual

        return out2


class MedGANDiscriminator(nn.Module):
    """Discriminator with minibatch averaging.

    Args:
        input_dim (int): Dimensionality of the input.
        hidden_dim (int): Width of hidden layers. Default: 256.
        minibatch_averaging (bool): Concatenate batch mean to each sample. Default: True.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 minibatch_averaging: bool = True):
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
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)
        return self.model(x)


def _weights_init(m):
    """Xavier uniform initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MedGAN(BaseModel):
    """MedGAN: Medical Generative Adversarial Network.

    Generates synthetic binary EHR records via a two-phase training process:
    (1) pre-train a linear autoencoder, then (2) adversarial training with
    standard BCE loss. The generator maps noise to the autoencoder's latent
    space, and the decoder projects back to binary medical codes.

    Reference:
        Choi et al., "Generating Multi-label Discrete Patient Records using
        Generative Adversarial Networks", MLHC 2017.

    Args:
        dataset (SampleDataset): A fitted SampleDataset with
            ``input_schema = {"visits": "multi_hot"}``.
        latent_dim (int): Dimensionality of the generator latent space. Default: 128.
        hidden_dim (int): Hidden layer width for the generator. Default: 128.
        autoencoder_hidden_dim (int): Autoencoder latent dimension. Default: 128.
        discriminator_hidden_dim (int): Discriminator hidden layer width. Default: 256.
        minibatch_averaging (bool): Use minibatch averaging in discriminator. Default: True.
        batch_size (int): Training batch size. Default: 512.
        ae_epochs (int): Autoencoder pre-training epochs. Default: 100.
        gan_epochs (int): Adversarial training epochs. Default: 200.
        ae_lr (float): Autoencoder learning rate. Default: 0.001.
        gan_lr (float): GAN learning rate. Default: 0.001.
        save_dir (str): Checkpoint save directory. Default: ``"./medgan_checkpoints"``.
        **kwargs: Additional arguments passed to ``BaseModel``.

    Examples:
        >>> from pyhealth.datasets.sample_dataset import InMemorySampleDataset
        >>> dataset = InMemorySampleDataset(
        ...     samples=[
        ...         {"patient_id": "p1", "visits": ["A", "B", "C"]},
        ...         {"patient_id": "p2", "visits": ["A", "C", "D"]},
        ...     ],
        ...     input_schema={"visits": "multi_hot"},
        ...     output_schema={},
        ... )
        >>> model = MedGAN(dataset, latent_dim=32, hidden_dim=32)
        >>> isinstance(model, MedGAN)
        True
    """

    def __init__(
        self,
        dataset,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        autoencoder_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        minibatch_averaging: bool = True,
        batch_size: int = 512,
        ae_epochs: int = 100,
        gan_epochs: int = 200,
        ae_lr: float = 0.001,
        gan_lr: float = 0.001,
        save_dir: str = "./medgan_checkpoints",
        **kwargs,
    ):
        super().__init__(dataset=dataset)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.ae_epochs = ae_epochs
        self.gan_epochs = gan_epochs
        self.ae_lr = ae_lr
        self.gan_lr = gan_lr
        self.save_dir = save_dir

        # Derive vocabulary size from processor
        processor = dataset.input_processors["visits"]
        self.input_dim = processor.size()

        # Build reverse lookup: index -> code string
        self._idx_to_code: List[Optional[str]] = [None] * self.input_dim
        for code, idx in processor.label_vocab.items():
            self._idx_to_code[idx] = code

        # Initialize components
        self.autoencoder = MedGANAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=autoencoder_hidden_dim,
        )
        self.generator = MedGANGenerator(
            latent_dim=latent_dim,
            hidden_dim=autoencoder_hidden_dim,
        )
        self.discriminator = MedGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging,
        )

        self.autoencoder.apply(_weights_init)
        self.generator.apply(_weights_init)
        self.discriminator.apply(_weights_init)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Not used in GAN context."""
        raise NotImplementedError(
            "Use train_model() for training and synthesize_dataset() for generation."
        )

    def train_model(self, train_dataset, val_dataset=None):
        """Train MedGAN on a SampleDataset.

        Phase 1: pre-train the autoencoder with BCE reconstruction loss.
        Phase 2: adversarial training with standard BCE GAN loss (not WGAN).

        Args:
            train_dataset: A fitted SampleDataset with
                ``input_schema = {"visits": "multi_hot"}``.
            val_dataset: Unused. Accepted for API compatibility.

        Returns:
            None
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Training MedGAN on: {device}")

        # Build multi-hot matrix from pre-encoded tensors
        tensors = [train_dataset[i]["visits"] for i in range(len(train_dataset))]
        data_matrix = torch.stack(tensors).numpy()

        medgan_ds = MedGANDataset(data=data_matrix)
        sampler = torch.utils.data.sampler.RandomSampler(
            data_source=medgan_ds, replacement=True,
        )
        dataloader = DataLoader(
            medgan_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        # ---- Phase 1: Autoencoder pretraining ----
        print(f"Phase 1: Pretraining autoencoder for {self.ae_epochs} epochs...")
        optimizer_ae = torch.optim.Adam(
            self.autoencoder.parameters(), lr=self.ae_lr,
        )
        criterion_ae = nn.BCELoss()

        self.autoencoder.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                real = batch.to(device)
                recon = self.autoencoder(real)
                loss = criterion_ae(recon, real)

                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, self.ae_epochs // 10) == 0 or epoch == 0:
                avg = total_loss / n_batches
                print(f"  AE epoch {epoch + 1}/{self.ae_epochs} loss={avg:.4f}")

        # ---- Phase 2: Adversarial training ----
        print(f"Phase 2: Adversarial training for {self.gan_epochs} epochs...")
        optimizer_g = torch.optim.Adam(
            list(self.generator.parameters())
            + list(self.autoencoder.decoder.parameters()),
            lr=self.gan_lr,
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.gan_lr,
        )

        best_d_loss = float("inf")

        for epoch in range(self.gan_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_batches = 0

            self.generator.train()
            self.discriminator.train()
            self.autoencoder.eval()
            self.autoencoder.decoder.train()

            for batch in dataloader:
                real = batch.to(device)
                bs = real.size(0)

                # --- Train Discriminator ---
                optimizer_d.zero_grad()
                noise = torch.randn(bs, self.latent_dim, device=device)
                fake_hidden = self.generator(noise)
                fake = self.autoencoder.decode(fake_hidden)

                real_pred = self.discriminator(real)
                fake_pred = self.discriminator(fake.detach())

                d_loss = (
                    F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
                    + F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
                )
                d_loss.backward()
                optimizer_d.step()

                # --- Train Generator ---
                optimizer_g.zero_grad()
                fake_pred = self.discriminator(fake)
                g_loss = F.binary_cross_entropy(
                    fake_pred, torch.ones_like(fake_pred),
                )
                g_loss.backward()
                optimizer_g.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                n_batches += 1

            avg_d = epoch_d_loss / n_batches
            avg_g = epoch_g_loss / n_batches

            if (epoch + 1) % max(1, self.gan_epochs // 10) == 0 or epoch == 0:
                print(
                    f"  GAN epoch {epoch + 1}/{self.gan_epochs} "
                    f"D_loss={avg_d:.4f} G_loss={avg_g:.4f}"
                )

            # Save best checkpoint
            if avg_d < best_d_loss:
                best_d_loss = avg_d
                self.save_model(os.path.join(self.save_dir, "best.pt"))

        # Save final checkpoint
        self.save_model(os.path.join(self.save_dir, "final.pt"))
        print("Training complete.")

    def synthesize_dataset(
        self, num_samples: int, random_sampling: bool = True,
    ) -> List[Dict]:
        """Generate synthetic patient records.

        Each synthetic patient is a flat list of ICD code strings decoded from
        a generated binary vector, matching the ``multi_hot`` input schema.

        Args:
            num_samples (int): Number of synthetic patients to generate.
            random_sampling (bool): Unused; accepted for API compatibility.

        Returns:
            list of dict: Synthetic patient records. Each dict has:
                ``"patient_id"`` (str): e.g. ``"synthetic_0"``.
                ``"visits"`` (list of str): flat list of decoded ICD code strings.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.generator.eval()
        self.autoencoder.eval()

        gen_samples = np.zeros((num_samples, self.input_dim), dtype=np.float32)
        n_full = num_samples // self.batch_size

        with torch.no_grad():
            for i in range(n_full):
                z = torch.randn(self.batch_size, self.latent_dim, device=device)
                fake = self.autoencoder.decode(self.generator(z))
                gen_samples[i * self.batch_size : (i + 1) * self.batch_size] = (
                    fake.cpu().numpy()
                )

            remaining = num_samples % self.batch_size
            if remaining > 0:
                z = torch.randn(remaining, self.latent_dim, device=device)
                fake = self.autoencoder.decode(self.generator(z))
                gen_samples[n_full * self.batch_size :] = fake.cpu().numpy()

        # Binarize at threshold 0.5
        gen_samples = (gen_samples >= 0.5).astype(np.float32)

        # Decode to code strings
        results: List[Dict] = []
        for i in range(num_samples):
            codes = [
                self._idx_to_code[idx]
                for idx in np.where(gen_samples[i] == 1.0)[0]
                if self._idx_to_code[idx] not in (None, "<pad>", "<unk>")
            ]
            results.append({
                "patient_id": f"synthetic_{i}",
                "visits": codes,
            })
        return results

    def save_model(self, path: str):
        """Save model weights to a checkpoint file.

        Args:
            path (str): File path to write the checkpoint.
        """
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

    def load_model(self, path: str):
        """Load model weights from a checkpoint file.

        Args:
            path (str): File path to read the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
