"""CorGAN: Correlation-capturing GAN for synthetic EHR generation.

This is a port of the reference implementation
(https://github.com/astorfi/cor-gan, specifically
``reference/cor-gan/Generative/corGAN/pytorch/CNN/MIMIC/wgancnnmimic.py``)
wrapped as a PyHealth ``BaseModel`` so it consumes the standard
``dataset -> SampleDataset -> model`` pipeline.

CorGAN treats each patient as a flat bag-of-codes (no visit structure), so it
expects an input feature named ``visits`` backed by a ``MultiHotProcessor``.
Training has two phases (mirroring the reference):

* a **convolutional autoencoder** is pre-trained with a sparse-friendly BCE
  reconstruction loss, then
* a **WGAN** adversarial phase runs the generator + decoder against a
  Lipschitz-clipped critic (no sigmoid; weight clipping in
  ``[clamp_lower, clamp_upper]``).

For tiny vocabularies that can't survive the 6-layer convolutional chain we
automatically fall back to the linear-autoencoder variant noted in the
reference's commented-out alternative. The public ``CorGAN`` class follows the
same API style as :class:`pyhealth.models.generators.HALO`
(``train_model`` / ``generate`` / ``save_model`` / ``load_model``).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from pyhealth.models import BaseModel


# ----------------------------------------------------------------------------
# Building blocks (ported from reference wgancnnmimic.py)
# ----------------------------------------------------------------------------
class _MultiHotDataset(Dataset):
    """Tiny ``torch.utils.data.Dataset`` over a multi-hot numpy matrix."""

    def __init__(self, data: np.ndarray):
        self.data = np.clip(data.astype(np.float32), 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class CorGANCNNAutoencoder(nn.Module):
    """1D-CNN autoencoder from the reference CorGAN paper.

    Six 1D conv layers compress the multi-hot vector down to a tiny latent;
    six transposed-conv layers project back. When ``use_adaptive_pooling`` is
    True we tack on an ``AdaptiveAvgPool1d`` so the decoder hits the exact
    vocabulary size for any input dim (the original CNN was hard-coded for
    MIMIC's vocabulary).

    Args:
        feature_size: Vocabulary size.
        use_adaptive_pooling: If True, force decoder output to ``feature_size``
            via adaptive average pooling. Default: True.
    """

    def __init__(self, feature_size: int, use_adaptive_pooling: bool = True):
        super().__init__()
        self.feature_size = feature_size
        self.use_adaptive_pooling = use_adaptive_pooling
        c = 4  # n_channels_base, per reference

        # Encoder: kernels (5,5,5,5,5,8), strides (2,2,3,3,3,1).
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(c, 2 * c, kernel_size=5, stride=2),
            nn.BatchNorm1d(2 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(2 * c, 4 * c, kernel_size=5, stride=3),
            nn.BatchNorm1d(4 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(4 * c, 8 * c, kernel_size=5, stride=3),
            nn.BatchNorm1d(8 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8 * c, 16 * c, kernel_size=5, stride=3),
            nn.BatchNorm1d(16 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16 * c, 32 * c, kernel_size=8, stride=1),
            nn.Tanh(),
        )

        # Decoder: kernels (5,5,7,7,7,3), strides (1,4,4,3,2,2). First layer
        # has NO BatchNorm, matching the reference. The original was hard-coded
        # for MIMIC's vocabulary; adaptive pooling rescales the output to any
        # feature_size for downstream Sigmoid binarisation.
        decoder_layers = [
            nn.ConvTranspose1d(32 * c, 16 * c, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16 * c, 8 * c, kernel_size=5, stride=4),
            nn.BatchNorm1d(8 * c),
            nn.ReLU(),
            nn.ConvTranspose1d(8 * c, 4 * c, kernel_size=7, stride=4),
            nn.BatchNorm1d(4 * c),
            nn.ReLU(),
            nn.ConvTranspose1d(4 * c, 2 * c, kernel_size=7, stride=3),
            nn.BatchNorm1d(2 * c),
            nn.ReLU(),
            nn.ConvTranspose1d(2 * c, c, kernel_size=7, stride=2),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.ConvTranspose1d(c, 1, kernel_size=3, stride=2),
        ]
        if use_adaptive_pooling:
            decoder_layers.append(nn.AdaptiveAvgPool1d(output_size=feature_size))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Allow either (B, F) or (B, 1, F) input.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        decoded = self.decoder(self.encoder(x))
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded

    def decode(self, latent):
        """Decode a latent emitted by the generator.

        The generator outputs ``(B, hidden_dim)``; we add a length-1 spatial
        axis so the transposed-conv stack accepts it.
        """
        if latent.dim() == 2:
            latent = latent.unsqueeze(2)
        decoded = self.decoder(latent)
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded


class CorGANLinearAutoencoder(nn.Module):
    """Linear autoencoder, the reference's commented-out alternative.

    Used as a fallback for small vocabularies where the 6-layer CNN can't
    physically compress the input (its smallest viable input is ~500
    features). For unordered code spaces this is often a stronger baseline
    anyway, since 1D conv assumes spatial locality.
    """

    def __init__(self, feature_size: int, latent_dim: int = 128):
        super().__init__()
        self.feature_size = feature_size
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def decode(self, latent):
        return self.decoder(latent)


class CorGANGenerator(nn.Module):
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


class CorGANCritic(nn.Module):
    """4-layer MLP critic with optional minibatch averaging.

    No sigmoid at the output: this is a Wasserstein critic (not a
    classifier), per the reference WGAN training loop.
    """

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
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if self.minibatch_averaging:
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)
        return self.model(x)


def _weights_init(m):
    """Reference initialization scheme (wgancnnmimic.py)."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _autoencoder_loss(x_output, y_target):
    """Sparse-friendly BCE used by the reference CorGAN autoencoder.

    Sum over features, then mean over batch -- equivalent to
    ``BCELoss(reduction='sum') / batch_size``. ``BCELoss(reduction='mean')``
    additionally means over features and dilutes the signal for sparse
    multi-hot targets.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (
        1.0 - y_target
    ) * torch.log(1.0 - x_output + epsilon)
    return torch.mean(-torch.sum(term, dim=1), dim=0)


# Minimum feature size the 6-layer CNN encoder can survive without producing a
# non-positive spatial dimension. The reference targets MIMIC's ~7k vocabulary;
# the conv chain (kernels 5,5,5,5,5,8 / strides 2,2,3,3,3,1) only produces a
# positive output for inputs >= ~1000. We pick a slightly conservative floor.
_CNN_MIN_FEATURES = 1000

# The reference CNN autoencoder's bottleneck has 32 * n_channels_base (= 32*4)
# output channels. The generator output is fed in as those channels, so this
# is fixed by architecture.
_CNN_BOTTLENECK_DIM = 32 * 4


# ----------------------------------------------------------------------------
# PyHealth BaseModel wrapper
# ----------------------------------------------------------------------------
class CorGAN(BaseModel):
    """CorGAN synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Trains a 1D-convolutional autoencoder + WGAN generator/critic on multi-hot
    patient vectors and generates new synthetic patients by sampling noise,
    pushing it through the generator, and decoding back with the (jointly
    trained) decoder.

    Generation is **unconditional**: each synthetic patient is a flat bag of
    codes (no visit structure), matching the ``multi_hot`` input schema.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": "multi_hot"}`` and whose ``output_schema`` is empty.
        latent_dim: Generator noise dimensionality. Default: 128.
        hidden_dim: Generator hidden width. Default: 128.
        discriminator_hidden_dim: Critic hidden width. Default: 256.
        minibatch_averaging: Concatenate per-batch mean to each critic input.
            Default: True.
        autoencoder_type: One of ``"cnn"`` (the reference) or ``"linear"``
            (the reference's commented-out alternative). For vocabularies
            smaller than ~500 the CNN cannot compress the input, so we
            silently fall back to ``"linear"``. Default: ``"cnn"``.
        use_adaptive_pooling: When using the CNN autoencoder, add an
            ``AdaptiveAvgPool1d`` so the decoder matches any vocabulary size.
            Ignored for the linear variant. Default: True.
        batch_size: Training batch size. Default: 512.
        ae_epochs: Autoencoder pre-training epochs. Default: 100.
        gan_epochs: Adversarial training epochs. Default: 200.
        lr: Learning rate for all optimizers. Default: 1e-3.
        weight_decay: L2 regularisation for all Adam optimizers. Default: 1e-4.
        b1: Adam beta1. Default: 0.9.
        b2: Adam beta2. Default: 0.999.
        n_iter_D: Critic updates per generator update (reference: 5).
        clamp_lower: WGAN critic weight-clip lower bound. Default: -0.01.
        clamp_upper: WGAN critic weight-clip upper bound. Default:  0.01.
        save_dir: Checkpoint directory used by ``train_model``.
            Default: ``"./save/corgan/"``.

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
        >>> model = CorGAN(dataset, latent_dim=16, hidden_dim=16, batch_size=2)
        >>> isinstance(model, CorGAN)
        True
    """

    def __init__(
        self,
        dataset,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        minibatch_averaging: bool = True,
        autoencoder_type: str = "cnn",
        use_adaptive_pooling: bool = True,
        batch_size: int = 512,
        ae_epochs: int = 100,
        gan_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        n_iter_D: int = 5,
        clamp_lower: float = -0.01,
        clamp_upper: float = 0.01,
        save_dir: str = "./save/corgan/",
    ) -> None:
        super().__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "CorGAN expects an input feature named 'visits' backed by a "
                "MultiHotProcessor."
            )

        self._batch_size = batch_size
        self._ae_epochs = ae_epochs
        self._gan_epochs = gan_epochs
        self._lr = lr
        self._weight_decay = weight_decay
        self._betas = (b1, b2)
        self._n_iter_D = n_iter_D
        self._clamp_lower = clamp_lower
        self._clamp_upper = clamp_upper
        self.save_dir = save_dir

        # Code vocab from the MultiHotProcessor's label_vocab.
        self.visits_processor = dataset.input_processors["visits"]
        self.input_dim = self.visits_processor.size()
        self._idx_to_code: List[Optional[str]] = [None] * self.input_dim
        for code, idx in self.visits_processor.label_vocab.items():
            self._idx_to_code[idx] = code

        # CNN can't compress small vocabularies; fall back to linear.
        if autoencoder_type == "cnn" and self.input_dim < _CNN_MIN_FEATURES:
            autoencoder_type = "linear"
        self.autoencoder_type = autoencoder_type

        if autoencoder_type == "linear":
            # Linear AE: bottleneck = generator hidden dim (user-controlled).
            self.autoencoder = CorGANLinearAutoencoder(
                feature_size=self.input_dim, latent_dim=hidden_dim
            )
        elif autoencoder_type == "cnn":
            # CNN AE: bottleneck is fixed at 128 by the conv-channel ladder.
            # The generator must emit that many features so the transposed-conv
            # decoder accepts its output. We silently align hidden_dim to the
            # CNN bottleneck (the reference always uses 128).
            if hidden_dim != _CNN_BOTTLENECK_DIM:
                hidden_dim = _CNN_BOTTLENECK_DIM
            self.autoencoder = CorGANCNNAutoencoder(
                feature_size=self.input_dim,
                use_adaptive_pooling=use_adaptive_pooling,
            )
        else:
            raise ValueError(
                f"Unknown autoencoder_type={autoencoder_type!r}; "
                "expected 'cnn' or 'linear'."
            )

        # The generator's residual connection requires latent_dim == hidden_dim
        # (per the reference). Align silently if the user mismatched.
        if latent_dim != hidden_dim:
            latent_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.generator = CorGANGenerator(
            latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.critic = CorGANCritic(
            input_dim=self.input_dim,
            hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging,
        )

        self.autoencoder.apply(_weights_init)
        self.generator.apply(_weights_init)
        self.critic.apply(_weights_init)

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """CorGAN does not have a single supervised forward pass.

        Use :meth:`train_model` for training and :meth:`generate` for
        synthesis. ``forward`` is implemented only to satisfy the
        ``BaseModel`` abstract contract.
        """
        raise NotImplementedError(
            "CorGAN is a GAN: use train_model() and generate() instead of "
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
        """Stack the multi-hot tensors of ``dataset`` into a DataLoader."""
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

    def train_model(self, train_dataset, val_dataset=None, device=None) -> Dict:
        """Train CorGAN with a custom two-phase loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Phase 1 pre-trains the autoencoder with
        sparse BCE reconstruction loss; phase 2 runs WGAN adversarial
        training (weight-clipped critic, joint generator + decoder).

        Args:
            train_dataset: ``SampleDataset`` for training.
            val_dataset: Unused; accepted for API symmetry.
            device: Device to train on. If ``None``, uses CUDA when available.

        Returns:
            Dict with keys ``"autoencoder_loss"``, ``"critic_loss"``,
            ``"generator_loss"`` -- one float per epoch in each list.
        """
        device = self._resolve_device(device)
        self.to(device)
        print(f"Training CorGAN on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        dataloader = self._build_dataloader(train_dataset)
        history: Dict[str, List[float]] = {
            "autoencoder_loss": [],
            "critic_loss": [],
            "generator_loss": [],
        }

        # ---- Phase 1: Autoencoder pretraining ----
        optimizer_ae = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        for epoch in tqdm(range(self._ae_epochs), desc="AE pretrain"):
            self.autoencoder.train()
            last_loss = 0.0
            for batch in dataloader:
                real = batch.to(self.device)
                recon = self.autoencoder(real)
                loss = _autoencoder_loss(recon, real)

                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()
                last_loss = loss.item()
            history["autoencoder_loss"].append(last_loss)

        # ---- Phase 2: WGAN adversarial training ----
        # The reference jointly optimises the generator and the autoencoder's
        # decoder, with a smaller LR on the decoder. We reuse that scheme.
        g_params = [
            {"params": self.generator.parameters()},
            {"params": self.autoencoder.decoder.parameters(), "lr": 1e-4},
        ]
        optimizer_g = torch.optim.Adam(
            g_params,
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        optimizer_d = torch.optim.Adam(
            self.critic.parameters(),
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        one = torch.tensor(1.0, device=self.device)
        mone = torch.tensor(-1.0, device=self.device)
        gen_iters = 0

        for epoch in tqdm(range(self._gan_epochs), desc="GAN train"):
            self.generator.train()
            self.critic.train()
            self.autoencoder.eval()
            self.autoencoder.decoder.train()

            last_d, last_g = 0.0, 0.0
            for real in dataloader:
                real = real.to(self.device)
                bs = real.size(0)

                # --- Train critic ---
                for p in self.critic.parameters():
                    p.requires_grad = True
                # Reference: ramp up critic iterations at the start and at
                # periodic intervals to keep the Wasserstein estimate tight.
                n_iter_D = (
                    100 if (gen_iters < 25 or gen_iters % 500 == 0)
                    else self._n_iter_D
                )
                for _ in range(n_iter_D):
                    for p in self.critic.parameters():
                        p.data.clamp_(self._clamp_lower, self._clamp_upper)

                    optimizer_d.zero_grad()
                    errD_real = torch.mean(self.critic(real)).squeeze()
                    errD_real.backward(one)

                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    fake = self.autoencoder.decode(self.generator(z))
                    errD_fake = torch.mean(self.critic(fake.detach())).squeeze()
                    errD_fake.backward(mone)
                    last_d = (errD_real - errD_fake).item()

                    optimizer_d.step()

                # --- Train generator ---
                for p in self.critic.parameters():
                    p.requires_grad = False
                optimizer_g.zero_grad()
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake = self.autoencoder.decode(self.generator(z))
                errG = torch.mean(self.critic(fake)).squeeze()
                errG.backward(one)
                optimizer_g.step()
                last_g = errG.item()
                gen_iters += 1

            history["critic_loss"].append(last_d)
            history["generator_loss"].append(last_g)

        self.save_model(os.path.join(self.save_dir, "final.pt"))
        return history

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
            HALO's nested-list output structure). CorGAN is a bag-of-codes
            model -- following the reference preprocessing, each patient is
            represented by the union of codes across all of their
            historical visits -- so the single inner list is that aggregate
            bag. The inner list may be empty if the generator produced an
            all-zero vector.
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
            # CorGAN models the patient as one aggregate bag of codes.
            results.append({"patient_id": f"synthetic_{i}", "visits": [codes]})
        return results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        """Save weights, vocabulary, and architecture metadata."""
        torch.save(
            {
                "autoencoder": self.autoencoder.state_dict(),
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "autoencoder_type": self.autoencoder_type,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "idx_to_code": self._idx_to_code,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load weights and vocabulary from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.critic.load_state_dict(ckpt["critic"])
        if "idx_to_code" in ckpt:
            self._idx_to_code = ckpt["idx_to_code"]
