"""TarDiff: target-oriented diffusion for synthetic clinical time series.

Implements Deng et al., "TarDiff: Target-Oriented Diffusion Guidance for
Synthetic Electronic Health Record Time Series Generation" (arXiv:2504.17613,
2025), rebuilt from the paper. The authors' code
(https://github.com/devin-21/TarDiff) has no license, so it was read but not
copied.

TarDiff has two parts:

* a **class-conditional diffusion model** on raw multivariate series
  ``(channels, length)``: a 1D U-Net that predicts the noise (or ``v``,
  Salimans and Ho, 2022), trained with the usual denoising loss (Ho et al., 2020) and sampled with DDIM (Song et
  al., 2021); and
* **influence guidance**: during sampling, each step is pushed towards
  series that would most reduce a trained downstream model's loss on a
  guidance set. With ``G`` the gradient of that loss with respect to the
  downstream model's parameters ``phi``, the influence of a sample is
  approximated to first order by ``G . grad_phi loss(x_t, y)`` (gradient
  alignment, Charpiat et al., 2019), and the step's mean moves by
  ``w * grad_x (G . grad_phi loss(x_t, y))`` (the paper's eq. for the guided
  mean).

The public :class:`TarDiff` class follows the API of the other generators
(``train_model`` / ``generate`` / ``save_model`` / ``load_model``). Any
PyHealth classifier trained on the same task can be the downstream model.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of diffusion steps.

    Args:
        t: Diffusion steps, ``(batch,)``.
        dim: Embedding size.

    Returns:
        ``(batch, dim)`` embeddings: cosines then sines of ``t`` at
        geometrically spaced frequencies (zero-padded when ``dim`` is odd).

    Examples:
        >>> import torch
        >>> from pyhealth.models.generators.tardiff import timestep_embedding
        >>> timestep_embedding(torch.tensor([0, 10]), 4).shape
        torch.Size([2, 4])
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, channels), channels)


class _ResBlock(nn.Module):
    """Residual block whose GroupNorm is scaled and shifted by the embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[..., None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class _Attention(nn.Module):
    """Multi-head self-attention over the time axis."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.norm = _norm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class UNet1D(nn.Module):
    """1D U-Net denoiser of a class-conditional diffusion model.

    It predicts the noise or ``v``, depending on :class:`TarDiff`'s
    ``prediction``.

    The defaults are the paper's: 64 base channels, multipliers ``(1, 2, 4,
    4)``, 3 residual blocks per level, 8-head attention at the first three
    levels (downsampling factors 1, 2 and 4) and a 32-dimensional class
    embedding. The length must be divisible by ``2 ** (len(mults) - 1)``.

    Args:
        in_channels: Channels (leads) of the series.
        n_classes: Number of classes to condition on.
        base_channels: Channels of the first level.
        mults: Channel multiplier of each level.
        num_res_blocks: Residual blocks per level.
        attention_levels: Levels (0 = full resolution) with self-attention.
        heads: Attention heads.
        class_dim: Size of the class embedding.
        dropout: Dropout inside the residual blocks.

    Examples:
        >>> import torch
        >>> from pyhealth.models.generators.tardiff import UNet1D
        >>> net = UNet1D(in_channels=3, n_classes=2, base_channels=8, mults=(1, 2),
        ...              num_res_blocks=1, attention_levels=(1,), heads=2)
        >>> net(torch.randn(4, 3, 16), torch.tensor([0, 1, 2, 3]),
        ...     torch.tensor([0, 1, 0, 1])).shape
        torch.Size([4, 3, 16])
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        base_channels: int = 64,
        mults: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        attention_levels: Sequence[int] = (0, 1, 2),
        heads: int = 8,
        class_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.downsample_factor = 2 ** (len(mults) - 1)
        emb_dim = 4 * base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(n_classes, class_dim)
        self.class_proj = nn.Linear(class_dim, emb_dim)
        self.inp = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        skips = [base_channels]
        ch = base_channels
        for level, mult in enumerate(mults):
            for _ in range(num_res_blocks):
                block = nn.ModuleList(
                    [_ResBlock(ch, base_channels * mult, emb_dim, dropout)]
                )
                ch = base_channels * mult
                if level in attention_levels:
                    block.append(_Attention(ch, heads))
                self.down.append(block)
                skips.append(ch)
            if level < len(mults) - 1:
                self.down.append(
                    nn.ModuleList([nn.Conv1d(ch, ch, 3, stride=2, padding=1)])
                )
                skips.append(ch)

        self.mid = nn.ModuleList(
            [
                _ResBlock(ch, ch, emb_dim, dropout),
                _Attention(ch, heads),
                _ResBlock(ch, ch, emb_dim, dropout),
            ]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for i in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [
                        _ResBlock(
                            ch + skips.pop(), base_channels * mult, emb_dim, dropout
                        )
                    ]
                )
                ch = base_channels * mult
                if level in attention_levels:
                    block.append(_Attention(ch, heads))
                if level > 0 and i == num_res_blocks:
                    block.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    block.append(nn.Conv1d(ch, ch, 3, padding=1))
                self.up.append(block)

        self.out = nn.Sequential(
            _norm(ch), nn.SiLU(), nn.Conv1d(ch, in_channels, 3, padding=1)
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the noise (or ``v``) of ``x`` ``(batch, channels, length)``
        at steps ``t`` for classes ``y``."""
        if x.shape[-1] % self.downsample_factor:
            raise ValueError(
                f"Length {x.shape[-1]} must be divisible by {self.downsample_factor}."
            )
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        emb = emb + self.class_proj(self.class_emb(y))
        h = self.inp(x)
        hs = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, _ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, _ResBlock) else layer(h)
        for block in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, _ResBlock) else layer(h)
        return self.out(h)


class TarDiff(BaseModel):
    """TarDiff generator: class-conditional diffusion with influence guidance.

    Trained on a dataset with one ``tensor`` input of shape ``(channels,
    length)`` and one ``binary`` or ``multiclass`` label. :meth:`train_model`
    fits the diffusion model; :meth:`generate` samples synthetic series for
    given classes, optionally guided by a downstream classifier through
    :meth:`guidance_gradient`.

    Choices the paper leaves open, made here: ``n_steps=1000`` diffusion steps
    with a linear beta schedule from 1e-4 to 0.02 (the paper gives no T),
    DDIM sampling, and ``G`` as the *mean* gradient over the guidance set
    (the text sums over the guidance set; Algorithm 1 divides by the size of
    the training set). The paper's guidance scale is 100 in its appendix,
    but its scale sweep peaks near 1000; tune ``guidance_scale`` on
    validation data.

    Args:
        dataset: The training :class:`~pyhealth.datasets.SampleDataset`.
        base_channels: U-Net channels of the first level. Default 64.
        mults: U-Net channel multipliers. Default ``(1, 2, 4, 4)``.
        num_res_blocks: Residual blocks per U-Net level. Default 3.
        attention_levels: U-Net levels with self-attention. Default ``(0, 1, 2)``.
        heads: Attention heads. Default 8.
        class_dim: Class embedding size. Default 32.
        dropout: Dropout in the residual blocks. Default 0.
        n_steps: Diffusion steps T. Default 1000.
        beta_start: First beta of the schedule. Default 1e-4.
        beta_end: Last beta of the schedule. Default 0.02.
        beta_schedule: ``"linear"`` spaces the betas evenly (DDPM).
            ``"scaled_linear"`` spaces their square roots evenly, as Latent
            Diffusion, which TarDiff's official code builds on, does. That
            keeps more of the signal at mid and late steps; the official
            code pairs it with ``n_steps=200``, ``beta_start=5e-4`` and
            ``beta_end=0.1``. Default ``"linear"``.
        prediction: What the network predicts. ``"eps"`` is the noise
            (DDPM, the paper, and the official code). ``"v"`` is
            ``sqrt(abar) * noise - sqrt(1 - abar) * x0`` (Salimans and Ho,
            2022), which stays accurate at the noisiest steps, where
            ``"eps"`` errors are amplified by ``1 / sqrt(abar)`` in the
            clean-series estimate. Default ``"eps"``.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TarDiff
        >>> samples = [
        ...     {"patient_id": str(i), "signal": torch.randn(2, 16), "label": i % 2}
        ...     for i in range(8)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = TarDiff(dataset, base_channels=8, mults=(1, 2), num_res_blocks=1,
        ...                 attention_levels=(), heads=2, n_steps=10)
        >>> _ = model.train_model(get_dataloader(dataset, batch_size=4), iterations=2)
        >>> out = model.generate(torch.tensor([0, 1]), steps=5)
        >>> out["signal"].shape, out["label"].tolist()
        (torch.Size([2, 2, 16]), [0, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        base_channels: int = 64,
        mults: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        attention_levels: Sequence[int] = (0, 1, 2),
        heads: int = 8,
        class_dim: int = 32,
        dropout: float = 0.0,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction: str = "eps",
    ):
        super().__init__(dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError("TarDiff expects one signal input and one label.")
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        if self.mode not in ("binary", "multiclass"):
            raise ValueError(
                f"TarDiff needs a binary or multiclass label, got {self.mode}."
            )
        sample = dataset[0][self.feature_key]
        if sample.dim() != 2:
            raise ValueError(
                f"TarDiff expects {self.feature_key} of shape (channels, length), "
                f"got {tuple(sample.shape)}."
            )
        self.n_channels, self.length = sample.shape
        self.n_classes = 2 if self.mode == "binary" else self.label_processor.size()
        self.config = {
            "base_channels": base_channels,
            "mults": tuple(mults),
            "num_res_blocks": num_res_blocks,
            "attention_levels": tuple(attention_levels),
            "heads": heads,
            "class_dim": class_dim,
            "dropout": dropout,
            "n_steps": n_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "prediction": prediction,
        }
        self.net = UNet1D(
            self.n_channels,
            self.n_classes,
            base_channels,
            mults,
            num_res_blocks,
            attention_levels,
            heads,
            class_dim,
            dropout,
        )
        if self.length % self.net.downsample_factor:
            raise ValueError(
                f"Series length {self.length} must be divisible by "
                f"{self.net.downsample_factor} for mults={tuple(mults)}."
            )
        if prediction not in ("eps", "v"):
            raise ValueError(f"prediction must be 'eps' or 'v', got {prediction!r}.")
        self.prediction = prediction
        self.n_steps = n_steps
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        elif beta_schedule == "scaled_linear":
            betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, n_steps, dtype=torch.float64
                )
                ** 2
            )
        else:
            raise ValueError(
                f"beta_schedule must be 'linear' or 'scaled_linear', "
                f"got {beta_schedule!r}."
            )
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0).float())

    def _classes(self, y: torch.Tensor) -> torch.Tensor:
        """Turns processed labels into class indices ``(batch,)``."""
        y = y.to(self.device)
        if self.mode == "binary":
            return y.reshape(-1).round().long()
        return y.reshape(-1).long()

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Noises clean series ``x0`` to step ``t``."""
        a = self.alphas_cumprod[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def _eps_and_x0(
        self, x: torch.Tensor, t: int, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The network's noise and clean-series estimates at step ``t``."""
        a = self.alphas_cumprod[t]
        out = self.net(x, torch.full_like(y, t), y)
        if self.prediction == "v":
            eps = (1 - a).sqrt() * x + a.sqrt() * out
            x0 = a.sqrt() * x - (1 - a).sqrt() * out
            return eps, x0
        return out, (x - (1 - a).sqrt() * out) / a.sqrt()

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Denoising loss of a batch.

        Args:
            **kwargs: A batch with the signal and the label.

        Returns:
            ``{"loss": mean squared error of the prediction}`` (the noise,
            or ``v``; see ``prediction``).
        """
        x0 = kwargs[self.feature_key].to(self.device).float()
        y = self._classes(kwargs[self.label_key])
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        target = noise
        if self.prediction == "v":
            a = self.alphas_cumprod[t].view(-1, 1, 1)
            target = a.sqrt() * noise - (1 - a).sqrt() * x0
        return {"loss": F.mse_loss(self.net(self.q_sample(x0, t, noise), t, y), target)}

    def train_model(
        self,
        train_dataloader: DataLoader,
        iterations: int = 20000,
        lr: float = 1e-4,
        progress: bool = False,
    ) -> list[float]:
        """Trains the diffusion model with Adam (the paper: 20,000 iterations, 1e-4).

        Args:
            train_dataloader: Batches of the training set.
            iterations: Optimizer steps.
            lr: Learning rate.
            progress: Show a progress bar.

        Returns:
            The loss of every step.

        Raises:
            ValueError: If the dataloader yields no batch.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.train()
        losses: list[float] = []
        bar = tqdm(total=iterations, disable=not progress, desc="TarDiff")
        while len(losses) < iterations:
            done = len(losses)
            for batch in train_dataloader:
                loss = self(**batch)["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar.update(1)
                if len(losses) == iterations:
                    break
            if len(losses) == done:
                bar.close()
                raise ValueError("The training dataloader yielded no batch.")
        bar.close()
        self.eval()
        return losses

    @staticmethod
    def guidance_gradient(
        guide: BaseModel, dataloader: DataLoader
    ) -> list[torch.Tensor]:
        """Mean gradient ``G`` of a trained classifier's loss over a guidance set.

        Args:
            guide: The downstream model, trained on the same task.
            dataloader: The guidance set (the paper uses the validation split).

        Returns:
            One tensor per trainable parameter of ``guide``, in
            ``guide.parameters()`` order.
        """
        params = [p for p in guide.parameters() if p.requires_grad]
        total = [torch.zeros_like(p) for p in params]
        n = 0
        was_training = guide.training
        guide.eval()
        for batch in dataloader:
            size = len(batch[guide.label_keys[0]])
            grads = torch.autograd.grad(
                guide(**batch)["loss"], params, allow_unused=True
            )
            for acc, g in zip(total, grads):
                if g is not None:
                    acc += g.detach() * size
            n += size
        guide.train(was_training)
        return [g / max(n, 1) for g in total]

    def _influence_grad(
        self,
        guide: BaseModel,
        G: list[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        cosine: bool = False,
    ) -> torch.Tensor:
        """``grad_x (G . grad_phi loss(x, y))``, summed over the batch.

        With ``cosine``, the dot product is divided by both gradients' norms,
        as TarDiff's official code does: one cosine for the whole batch.
        """
        params = [p for p in guide.parameters() if p.requires_grad]
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            # Processed binary labels are (batch, 1) floats.
            labels = y.float().unsqueeze(-1) if self.mode == "binary" else y
            batch = {self.feature_key: x, self.label_key: labels}
            # The batch loss is a mean; scale back to a sum so that every
            # sample's term is its own influence.
            loss = guide(**batch)["loss"] * x.shape[0]
            grads = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=True
            )
            pairs = [(g, Gi) for g, Gi in zip(grads, G) if g is not None]
            dot = sum((g * Gi).sum() for g, Gi in pairs)
            if cosine:
                norm = sum((g**2).sum() for g, _ in pairs).sqrt()
                norm_G = sum((Gi**2).sum() for _, Gi in pairs).sqrt()
                dot = dot / (norm * norm_G + 1e-12)
            (grad_x,) = torch.autograd.grad(dot, x)
        return grad_x

    def class_indices(self, labels: Sequence[Any]) -> torch.Tensor:
        """Maps raw labels (as in the dataset, e.g. ``"MI"``) to the class
        indices that :meth:`generate` takes.

        Args:
            labels: Raw label values.

        Returns:
            ``(n,)`` long tensor of class indices.

        Raises:
            KeyError: If a label is not in the label processor's vocabulary.
        """
        vocab = self.label_processor.label_vocab
        return torch.tensor([vocab[label] for label in labels], dtype=torch.long)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        guide: BaseModel | None = None,
        guidance: list[torch.Tensor] | None = None,
        guidance_scale: float = 100.0,
        batch_size: int = 256,
        generator: torch.Generator | None = None,
        guide_on_x0: bool = False,
        cosine: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Samples one synthetic series per label with DDIM.

        Without ``guide`` this is plain class-conditional sampling. With a
        ``guide`` and its ``guidance`` (from :meth:`guidance_gradient`), every
        step's mean is moved by ``guidance_scale * grad_x (G . grad_phi
        loss(x_t, y))``.

        ``guide_on_x0`` and ``cosine`` switch to what TarDiff's official code
        does instead of the paper's equation: the guide scores the step's
        clean-series estimate ``x0`` (the kind of input it was trained on)
        rather than the noisy ``x_t``, and the dot product becomes a cosine
        over the batch, so it can't grow just by making the samples harder
        to classify. The official code also shrinks the step by
        ``1 - guidance_scale``, which this leaves out.

        Args:
            labels: Class indices to generate, ``(n,)``: the label
                processor's indices (``label_processor.label_vocab`` values),
                not raw labels; :meth:`class_indices` converts raw labels.
            steps: DDIM steps (evenly spaced over the ``n_steps``).
            eta: DDIM noise; 0 is deterministic, 1 is DDPM-like.
            guide: Trained downstream model for influence guidance.
            guidance: ``G`` for ``guide``. Required with ``guide``.
            guidance_scale: Guidance weight ``w``.
            batch_size: Series generated at once.
            generator: Random generator for reproducible samples (on the
                model's device).
            guide_on_x0: Score ``x0`` instead of ``x_t``. Default False.
            cosine: Use the cosine instead of the dot product. Default False.
                The cosine is taken over the whole batch, so the push on each
                series shrinks as ``batch_size`` grows; re-tune
                ``guidance_scale`` when changing it.

        Returns:
            ``{"signal": (n, channels, length), "label": (n,)}`` on the CPU,
            with class indices as labels (:meth:`to_samples` maps them back).
        """
        if guide is not None and guidance is None:
            raise ValueError("Pass the guide's guidance gradient (guidance_gradient).")
        self.eval()
        guide_was_training = guide is not None and guide.training
        if guide is not None:
            guide.eval()
            guidance = [g.to(self.device) for g in guidance]
        labels = torch.as_tensor(labels).reshape(-1).long()
        times = torch.linspace(self.n_steps - 1, 0, steps).round().long().tolist()
        out = []
        for start in range(0, len(labels), batch_size):
            y = labels[start : start + batch_size].to(self.device)
            x = torch.randn(
                len(y),
                self.n_channels,
                self.length,
                device=self.device,
                generator=generator,
            )
            for i, t in enumerate(times):
                a_t = self.alphas_cumprod[t]
                a_prev = (
                    self.alphas_cumprod[times[i + 1]]
                    if i + 1 < len(times)
                    else a_t.new_tensor(1.0)
                )
                eps, x0 = self._eps_and_x0(x, t, y)
                sigma = (
                    eta
                    * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                    .clamp(min=0)
                    .sqrt()
                )
                mean = (
                    a_prev.sqrt() * x0
                    + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
                )
                if guide is not None:
                    mean = mean + guidance_scale * self._influence_grad(
                        guide, guidance, x0 if guide_on_x0 else x, y, cosine
                    )
                noise = torch.randn(x.shape, device=self.device, generator=generator)
                x = mean + sigma * noise
            out.append(x.cpu())
        if guide is not None:
            guide.train(guide_was_training)
        return {"signal": torch.cat(out), "label": labels}

    def to_samples(
        self, generated: dict[str, torch.Tensor], prefix: str = "synthetic"
    ) -> list[dict]:
        """Turns :meth:`generate` output into samples for ``create_sample_dataset``.

        Args:
            generated: The output of :meth:`generate`.
            prefix: Patient id prefix of the synthetic samples.

        Returns:
            One dict per series with ``patient_id``, the signal and the raw
            label (the class index mapped back through the label processor's
            vocabulary), under the training dataset's keys.
        """
        raw = {
            index: label for label, index in self.label_processor.label_vocab.items()
        }
        return [
            {
                "patient_id": f"{prefix}-{i}",
                self.feature_key: x,
                self.label_key: raw[int(y)],
            }
            for i, (x, y) in enumerate(zip(generated["signal"], generated["label"]))
        ]

    def save_model(self, path: str) -> None:
        """Saves the weights and the configuration to ``path``."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    def load_model(self, path: str) -> None:
        """Loads weights saved by :meth:`save_model` into this model.

        Raises:
            ValueError: If the saved configuration differs from this model's.
        """
        saved = torch.load(path, map_location=self.device, weights_only=True)
        # Checkpoints from before these options existed used the defaults.
        saved["config"].setdefault("beta_schedule", "linear")
        saved["config"].setdefault("prediction", "eps")
        if saved["config"] != self.config:
            raise ValueError(
                f"Saved config {saved['config']} differs from {self.config}."
            )
        self.load_state_dict(saved["state_dict"])
