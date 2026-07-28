"""GPT-2 baseline for unconditional synthetic EHR generation.

A simple decoder-only baseline that mirrors the standalone reference script
``generate_synthetic_mimic3_gpt2.py`` (``--mode transformer_baseline``) but
plugged into the standard PyHealth ``dataset -> set_task -> SampleDataset ->
model`` pipeline. It consumes the same :class:`~pyhealth.tasks.EHRGeneration`
task as :class:`~pyhealth.models.HALO`.

Each patient's visits are flattened into a single token stream::

    [BOS]  <codes of visit 1>  [VISIT_DELIM]  <codes of visit 2>  ...  [EOS]

A small :class:`transformers.GPT2LMHeadModel` is trained on these streams with
causal language modeling. Generation autoregressively samples a token stream
(``do_sample`` + top-k/top-p) and decodes it back into per-visit code lists,
splitting on the ``[VISIT_DELIM]`` token.

The code vocabulary is taken from the dataset's ``NestedSequenceProcessor``
(which already reserves index 0 for ``<pad>`` and index 1 for ``<unk>``); three
special tokens (BOS, EOS, VISIT_DELIM) are appended, and ``<pad>`` (index 0) is
reused as the padding token.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel


class GPT2(BaseModel):
    """GPT-2 baseline synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": NestedSequenceProcessor}`` and whose ``output_schema``
            is empty.
        embed_dim: GPT-2 embedding dimension (``n_embd``). Must be divisible by
            ``n_heads``. Default: 512.
        n_heads: Number of attention heads. Default: 8.
        n_layers: Number of transformer layers. Default: 8.
        max_len: Maximum token-stream length (``n_positions``); streams are
            truncated to this length. Default: 512.
        batch_size: Training batch size. Default: 64.
        epochs: Number of training epochs. Default: 50.
        lr: Learning rate for the Adam optimizer. Default: 1e-4.
        save_dir: Directory for checkpoints written by ``train_model``.
            Default: ``"./save/"``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
        ...     {"patient_id": "p2", "visits": [["A"], ["B", "C"]]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "nested_sequence"},
        ...     output_schema={},
        ... )
        >>> model = GPT2(dataset, embed_dim=16, n_heads=2, n_layers=2, max_len=64)
        >>> isinstance(model, GPT2)
        True
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_len: int = 512,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-4,
        save_dir: str = "./save/",
    ) -> None:
        super(GPT2, self).__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "GPT2 expects an input feature named 'visits' backed by a "
                "NestedSequenceProcessor."
            )

        self.save_dir = save_dir
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr
        self.max_len = max_len

        # Code vocab from the NestedSequenceProcessor (includes <pad>=0, <unk>=1).
        self.visits_processor = dataset.input_processors["visits"]
        self.code_vocab_size = self.visits_processor.vocab_size()
        # Append three special tokens after the code vocab; reuse <pad>=0 as PAD.
        self.bos_id = self.code_vocab_size
        self.eos_id = self.code_vocab_size + 1
        self.delim_id = self.code_vocab_size + 2
        self.pad_id = 0
        total_vocab_size = self.code_vocab_size + 3

        config = GPT2Config(
            vocab_size=total_vocab_size,
            n_positions=max_len,
            n_embd=embed_dim,
            n_layer=n_layers,
            n_head=n_heads,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
        )
        # Registered as a sub-module so .parameters()/.to() work.
        self.gpt2 = GPT2LMHeadModel(config)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        """Resolve a user-supplied device, defaulting to CUDA when available."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Visit index tensor -> flat causal-LM token stream
    # ------------------------------------------------------------------
    def _encode_visits(self, visits: torch.Tensor):
        """Flatten the padded visit-index tensor into causal-LM token streams.

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)`` from
                the ``NestedSequenceProcessor``. Index 0 is ``<pad>`` and is
                skipped.

        Returns:
            input_ids: LongTensor ``(batch, L)`` token streams, right-padded.
            attention_mask: LongTensor ``(batch, L)`` (1 for real tokens).
            labels: ``input_ids`` with padding positions set to ``-100`` so they
                are ignored by the cross-entropy loss.
        """
        batch_seqs: List[List[int]] = []
        for i in range(visits.shape[0]):
            n_visits = int((visits[i].sum(dim=-1) > 0).sum().item())
            seq: List[int] = [self.bos_id]
            for j in range(n_visits):
                codes = [int(c) for c in visits[i, j].tolist() if c > 0]
                seq.extend(codes)
                if j < n_visits - 1:
                    seq.append(self.delim_id)
            seq.append(self.eos_id)
            batch_seqs.append(seq[: self.max_len])

        length = max(len(s) for s in batch_seqs)
        input_ids = torch.full(
            (len(batch_seqs), length), self.pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(batch_seqs), length), dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(batch_seqs):
            input_ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            attention_mask[i, : len(seq)] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, visits: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)`` from
                the ``NestedSequenceProcessor``.
            **kwargs: Any other batch keys are ignored.

        Returns:
            Dict with ``loss`` (scalar causal-LM cross-entropy) and ``y_prob``
            (next-token probabilities, shape ``(batch, L, vocab_size)``).
        """
        visits = visits.to(self.device)
        input_ids, attention_mask, labels = self._encode_visits(visits)
        out = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"loss": out.loss, "y_prob": F.softmax(out.logits, dim=-1)}

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------
    def train_model(self, train_dataset, val_dataset=None, device=None) -> None:
        """Train the GPT-2 baseline with a custom loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Uses the standard ``get_dataloader``, an Adam
        optimizer, and causal-LM loss. When ``val_dataset`` is given, validation
        loss is computed after each epoch and the best checkpoint is saved to
        ``self.save_dir``.

        Args:
            train_dataset: ``SampleDataset`` for training.
            val_dataset: Optional ``SampleDataset`` for validation.
            device: Device to train on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.
        """
        device = self._resolve_device(device)
        self.to(device)
        print(f"Training on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.gpt2.parameters(), lr=self._lr)

        checkpoint_path = os.path.join(self.save_dir, "gpt2_model")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.gpt2.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_loader = get_dataloader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        global_loss = 1e10
        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            self.gpt2.train()
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in batch_iter:
                visits = batch["visits"].to(self.device)
                input_ids, attention_mask, labels = self._encode_visits(visits)

                optimizer.zero_grad()
                out = self.gpt2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                out.loss.backward()
                optimizer.step()
                batch_iter.set_postfix(loss=f"{out.loss.item():.4f}")

            if val_dataset is not None:
                self.gpt2.eval()
                val_loader = get_dataloader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        visits = val_batch["visits"].to(self.device)
                        input_ids, attention_mask, labels = self._encode_visits(visits)
                        out = self.gpt2(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        val_losses.append(out.loss.item())

                cur_val_loss = float(np.mean(val_losses))
                print(f"Epoch {epoch} Validation Loss: {cur_val_loss:.7f}")
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        "model": self.gpt2.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(state, checkpoint_path)
                    print("------------ Save best model ------------")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _decode_ids(self, ids: List[int], index_to_code: Dict[int, str]) -> List[List[str]]:
        """Decode a generated token stream into per-visit code lists."""
        visits_out: List[List[str]] = []
        current: List[str] = []
        for tid in ids:
            if tid in (self.bos_id, self.pad_id):
                continue
            if tid == self.eos_id:
                break
            if tid == self.delim_id:
                if current:
                    visits_out.append(current)
                    current = []
                continue
            if tid < self.code_vocab_size:
                code = index_to_code.get(int(tid))
                if code not in (None, "<pad>", "<unk>"):
                    current.append(code)
        if current:
            visits_out.append(current)
        return visits_out

    def generate(
        self,
        num_samples: int,
        device=None,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[Dict]:
        """Generate synthetic patients with the trained GPT-2 baseline.

        Feeds a ``[BOS]`` token and autoregressively samples a token stream with
        ``top_k``/``top_p`` sampling, then decodes it into per-visit code lists.

        Args:
            num_samples: Number of synthetic patients to generate.
            device: Device to generate on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.
            top_k: Top-k sampling cutoff. Default: 50.
            top_p: Nucleus (top-p) sampling cutoff. Default: 0.95.

        Returns:
            List of dicts, each ``{"patient_id": "synthetic_i",
            "visits": [[code, ...], ...]}`` with decoded code strings.
        """
        device = self._resolve_device(device)
        self.to(device)

        index_to_code = {v: k for k, v in self.visits_processor.code_vocab.items()}

        self.gpt2.eval()
        synthetic_dataset: List[Dict] = []
        sample_batch_size = min(num_samples, 256)
        generated = 0
        pbar = tqdm(total=num_samples, desc="Generating patients")

        with torch.no_grad():
            while generated < num_samples:
                bs = min(sample_batch_size, num_samples - generated)
                input_ids = torch.full(
                    (bs, 1), self.bos_id, dtype=torch.long, device=self.device
                )
                out_ids = self.gpt2.generate(
                    input_ids,
                    max_length=self.max_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                )
                for i in range(bs):
                    visits_out = self._decode_ids(
                        out_ids[i].tolist(), index_to_code
                    )
                    synthetic_dataset.append(
                        {
                            "patient_id": f"synthetic_{generated + i}",
                            "visits": visits_out,
                        }
                    )
                generated += bs
                pbar.update(bs)
            pbar.close()

        return synthetic_dataset
