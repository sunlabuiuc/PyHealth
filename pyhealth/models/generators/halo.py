import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional

from pyhealth.models import BaseModel

from pyhealth.models.generators.halo_resources.halo_model import HALOModel
from pyhealth.models.generators.halo_resources.halo_config import HALOConfig


def _halo_collate_fn(batch):
    """Collate HALO samples, padding the visit dimension across the batch."""
    visits = pad_sequence(
        [item["visits"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    collated = {k: v for k, v in batch[0].items() if k != "visits"}
    collated["visits"] = visits
    return collated


class HALO(BaseModel):
    """HALO: Heterogeneous Autoregressive Language mOdel for synthetic EHR generation.

    Trains a GPT-2-style transformer on patient visit sequences and generates
    synthetic patients by autoregressive sampling.

    Args:
        dataset (SampleDataset): A SampleDataset whose input_schema contains
            ``{"visits": "nested_sequence"}`` and whose output_schema is empty.
        embed_dim: Transformer embedding dimension. Default: 768.
        n_heads: Number of attention heads. Default: 12.
        n_layers: Number of transformer layers. Default: 12.
        n_ctx: Maximum number of visits (context length). Default: 48.
        batch_size: Training batch size. Default: 48.
        epochs: Number of training epochs. Default: 50.
        pos_loss_weight: Positive-class weight for BCE loss. None means no
            weighting. Default: None.
        lr: Learning rate for Adam optimizer. Default: 1e-4.
        save_dir: Directory to save model checkpoints. Default: ``"./save/"``.

    Examples:
        >>> from pyhealth.datasets.sample_dataset import InMemorySampleDataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
        ...     {"patient_id": "p2", "visits": [["A"], ["B", "C"]]},
        ... ]
        >>> dataset = InMemorySampleDataset(
        ...     samples=samples,
        ...     input_schema={"visits": "nested_sequence"},
        ...     output_schema={},
        ... )
        >>> model = HALO(dataset, embed_dim=64, n_heads=2, n_layers=2, n_ctx=8)
        >>> isinstance(model, HALO)
        True
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_ctx: int = 48,
        batch_size: int = 48,
        epochs: int = 50,
        pos_loss_weight: Optional[float] = None,
        lr: float = 1e-4,
        save_dir: str = "./save/",
    ) -> None:
        super(HALO, self).__init__(dataset)

        self.save_dir = save_dir
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr

        # Derive vocab sizes from the dataset's NestedSequenceProcessor.
        visits_processor = dataset.input_processors["visits"]
        code_vocab_size = visits_processor.vocab_size()  # includes <pad> and <unk>
        label_vocab_size = 0  # generative task — no output labels
        # +3 special tokens: start-of-sequence, end-of-sequence, pad-visit
        total_vocab_size = code_vocab_size + label_vocab_size + 3

        self.config = HALOConfig(
            total_vocab_size=total_vocab_size,
            code_vocab_size=code_vocab_size,
            label_vocab_size=label_vocab_size,
            special_vocab_size=3,
            n_positions=n_ctx + 8,  # position embedding table; needs a bit of slack
            n_ctx=n_ctx,
            n_embd=embed_dim,
            n_layer=n_layers,
            n_head=n_heads,
            batch_size=batch_size,
            epoch=epochs,
            pos_loss_weight=pos_loss_weight,
            lr=lr,
        )

        # Store processor reference for use in synthesize_dataset.
        self.visits_processor = visits_processor

        # Register as an nn.Module sub-module so parameters() works correctly.
        self.halo_model = HALOModel(self.config)

    # ------------------------------------------------------------------
    # Multi-hot encoding helper
    # ------------------------------------------------------------------

    def _encode_visits(self, visits: torch.Tensor):
        """Convert a padded index tensor to HALO multi-hot format.

        The NestedSequenceProcessor returns indices; HALO's transformer expects
        multi-hot vectors of shape (batch, n_ctx, total_vocab_size).

        Args:
            visits: LongTensor of shape (batch, max_visits, max_codes_per_visit).
                Index 0 is the pad token and is skipped.

        Returns:
            batch_ehr: FloatTensor of shape ``(batch, n_ctx, total_vocab_size)``.
            batch_mask: FloatTensor of shape ``(batch, n_ctx-1, 1)``, shifted to
                align with the autoregressive prediction targets.
        """
        cfg = self.config
        batch_size = visits.shape[0]

        batch_ehr = torch.zeros(
            batch_size, cfg.n_ctx, cfg.total_vocab_size, device=self.device
        )
        batch_mask = torch.zeros(batch_size, cfg.n_ctx, 1, device=self.device)

        for i in range(batch_size):
            n_visits = min(visits.shape[1], cfg.n_ctx - 2)
            for j in range(n_visits):
                for code_idx in visits[i, j]:
                    if code_idx > 0:  # skip padding (index 0)
                        batch_ehr[i, j + 2, code_idx] = 1  # visits occupy positions 2+
                if visits[i, j].sum() > 0:
                    batch_mask[i, j + 2] = 1

            # Special tokens (label_vocab_size == 0, so the 3 extras are contiguous):
            batch_ehr[i, 0, cfg.code_vocab_size + cfg.label_vocab_size] = 1       # start
            batch_ehr[i, n_visits + 2, cfg.code_vocab_size + cfg.label_vocab_size + 1] = 1  # end
            batch_ehr[i, n_visits + 3:, cfg.code_vocab_size + cfg.label_vocab_size + 2] = 1  # pad

        batch_mask[:, 1] = 1           # label-position mask row
        batch_mask = batch_mask[:, 1:, :]  # shift mask to align with shifted labels/preds

        return batch_ehr, batch_mask

    # ------------------------------------------------------------------
    # forward — required by BaseModel (abstract in nn.Module)
    # ------------------------------------------------------------------

    def forward(self, visits: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Accepts the padded index tensor produced by NestedSequenceProcessor,
        converts it to HALO multi-hot format, and runs the transformer.

        Args:
            visits: LongTensor of shape ``(batch, max_visits, max_codes_per_visit)``.
            **kwargs: Additional keys from the batch dict are ignored.

        Returns:
            loss: scalar BCE loss tensor.
            predictions: code probability tensor of shape
                ``(batch, n_ctx, total_vocab_size)``.
        """
        visits = visits.to(self.device)
        batch_ehr, batch_mask = self._encode_visits(visits)

        loss, predictions, _ = self.halo_model(
            batch_ehr,
            position_ids=None,
            ehr_labels=batch_ehr,
            ehr_masks=batch_mask,
            pos_loss_weight=self.config.pos_loss_weight,
        )
        return {"loss": loss, "predictions": predictions}

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train_model(self, train_dataset, val_dataset=None) -> None:
        """Train the HALO model using a custom loop.

        Named ``train_model`` (not ``train``) to avoid shadowing ``nn.Module.train()``.

        Args:
            train_dataset: SampleDataset for training.
            val_dataset: Optional SampleDataset for validation. When provided,
                validation loss is computed after every epoch and the best
                checkpoint is saved to ``self.save_dir``.
        """
        # Move model to GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Training on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.halo_model.parameters(), lr=self._lr)

        checkpoint_path = os.path.join(self.save_dir, "halo_model")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.halo_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=False,  # IterableDataset (litdata.StreamingDataset) does not support shuffle
            drop_last=False,
            collate_fn=_halo_collate_fn,
        )

        global_loss = 1e10

        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            self.halo_model.train()
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in batch_iter:
                visits = batch["visits"].to(self.device)
                batch_ehr, batch_mask = self._encode_visits(visits)

                optimizer.zero_grad()
                loss, _, _ = self.halo_model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=self.config.pos_loss_weight,
                )
                loss.backward()
                optimizer.step()
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            if val_dataset is not None:
                self.halo_model.eval()
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self._batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=_halo_collate_fn,
                )
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        visits = val_batch["visits"].to(self.device)
                        batch_ehr, batch_mask = self._encode_visits(visits)
                        val_loss, _, _ = self.halo_model(
                            batch_ehr,
                            position_ids=None,
                            ehr_labels=batch_ehr,
                            ehr_masks=batch_mask,
                            pos_loss_weight=self.config.pos_loss_weight,
                        )
                        val_losses.append(val_loss.item())

                cur_val_loss = float(np.mean(val_losses))
                print(f"Epoch {epoch} Validation Loss: {cur_val_loss:.7f}")
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        "model": self.halo_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(state, checkpoint_path)
                    print("------------ Save best model ------------")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize_dataset(
        self, num_samples: int, random_sampling: bool = True
    ) -> List[Dict]:
        """Generate synthetic patients using the trained HALO model.

        Autoregressive sampling: feeds a start token and iteratively calls
        ``halo_model.sample()`` until an end token is produced or ``n_ctx``
        steps are reached.

        Args:
            num_samples: Number of synthetic patients to generate.
            random_sampling: If True, samples via Bernoulli (stochastic).
                If False, uses rounding (deterministic). Default: True.

        Returns:
            list of dict: Synthetic patient records. Each dict has two keys:
                ``"patient_id"`` (str): unique identifier, e.g. ``"synthetic_0"``.
                ``"visits"`` (list of list of str): decoded code strings per visit.
        """
        # Ensure model is on GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        cfg = self.config
        # Invert vocabulary: index → code string
        index_to_code = {v: k for k, v in self.visits_processor.code_vocab.items()}

        end_token_idx = cfg.code_vocab_size + cfg.label_vocab_size + 1

        # Build the start-token vector
        stoken = torch.zeros(cfg.total_vocab_size, device=self.device, dtype=torch.float32)
        stoken[cfg.code_vocab_size + cfg.label_vocab_size] = 1  # start token

        self.halo_model.eval()
        synthetic_dataset = []
        sample_batch_size = min(num_samples, 256)
        generated = 0

        with torch.no_grad():
            while generated < num_samples:
                bs = min(sample_batch_size, num_samples - generated)
                # prev: (bs, 1, total_vocab_size) — starts with just the start token
                prev = stoken.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
                empty = torch.zeros(
                    bs, 1, cfg.total_vocab_size, device=self.device, dtype=torch.float32
                )

                for _ in range(cfg.n_ctx - 1):
                    prev = self.halo_model.sample(
                        torch.cat((prev, empty), dim=1), random_sampling
                    )
                    # Early stop when all sequences have produced an end token
                    has_end = prev[:, :, end_token_idx].sum(dim=1).bool()
                    if has_end.all():
                        break

                batch_ehrs = prev.cpu().detach().numpy()

                for i in range(bs):
                    ehr = batch_ehrs[i]  # (seq_len, total_vocab_size)
                    visits_out = []
                    # Position 0 = start token; visits occupy positions 1+
                    for j in range(1, len(ehr)):
                        visit_row = ehr[j]
                        indices = np.nonzero(visit_row)[0]
                        visit_codes = []
                        hit_end = False
                        for idx in indices:
                            if idx < cfg.code_vocab_size:
                                code = index_to_code.get(idx)
                                if code not in (None, "<pad>", "<unk>"):
                                    visit_codes.append(code)
                            elif idx == end_token_idx:
                                hit_end = True
                        if visit_codes:
                            visits_out.append(visit_codes)
                        if hit_end:
                            break

                    synthetic_dataset.append(
                        {
                            "patient_id": f"synthetic_{generated + i}",
                            "visits": visits_out,
                        }
                    )
                generated += bs

        return synthetic_dataset
