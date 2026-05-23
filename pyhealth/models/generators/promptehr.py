"""PromptEHR: prompt-learning BART for synthetic EHR generation.

This is a PyHealth ``BaseModel`` port of PromptEHR (Wang & Sun, EMNLP'22,
https://github.com/RyanWangZf/PromptEHR), wrapped so it consumes the standard
``dataset -> set_task -> SampleDataset -> model`` pipeline and shares the same
:class:`~pyhealth.tasks.EHRGeneration` task as
:class:`~pyhealth.models.HALO` and :class:`~pyhealth.models.GPT2`.

PromptEHR treats sequential EHRs as a *neural database* and learns to fill in
patient records with a conditional **BART** (sequence-to-sequence denoising
autoencoder) trained with **prompt learning**. The three ideas that define the
reference implementation are preserved here:

* **BART seq2seq core.** Generation is encoder-decoder, not decoder-only. The
  reference subclasses ``BartForEHRSimulation`` from ``BartPretrainedModel``;
  this port wraps :class:`transformers.BartForConditionalGeneration`, mirroring
  the way :class:`~pyhealth.models.GPT2` wraps ``GPT2LMHeadModel``.
* **Prompt learning.** The reference reparameterizes a learnable prompt from
  patient baseline demographics and prepends it to the encoder/decoder
  (``ConditionalPrompt``). PyHealth's :class:`~pyhealth.tasks.EHRGeneration`
  task is *unconditional* (only ``visits``, no baseline features -- exactly like
  HALO/GPT2), so the prompt reduces to a learnable continuous **soft prefix**
  prepended to the encoder. This is the prompt-tuning core without the
  demographic reparameterization.
* **Span-infilling objective.** The reference learns by masking spans of codes
  and reconstructing them. Here the encoder sees a corrupted (randomly masked)
  copy of the patient's code stream and the decoder reconstructs the full
  stream -- the standard BART denoising objective and the same masked-prediction
  spirit.

Each patient's visits are serialized into a single code stream::

    [CODE_PROMPT]  <codes of visit 1>  [VISIT_DELIM]  <codes of visit 2>  ...  [EOS]

The reference handles several code types (diagnosis / procedure / drug / lab)
each with its own modality prompt token; the PyHealth ``EHRGeneration`` task
exposes a single ``visits`` modality, so a single ``[CODE_PROMPT]`` token marks
it. The code vocabulary is taken from the dataset's
``NestedSequenceProcessor`` (which already reserves index 0 for ``<pad>`` and
index 1 for ``<unk>``); five special tokens (BOS, EOS, VISIT_DELIM, MASK,
CODE_PROMPT) are appended, and ``<pad>`` (index 0) is reused as the pad token.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BartConfig, BartForConditionalGeneration

from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel


class PromptEHR(BaseModel):
    """PromptEHR synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Trains a BART denoising autoencoder with a learnable soft prompt on patient
    visit-code streams, then generates synthetic patients by prompt-conditioned
    encoder-decoder sampling. Generation is **unconditional** (no demographic
    conditioning), matching the :class:`~pyhealth.tasks.EHRGeneration` task.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": NestedSequenceProcessor}`` and whose ``output_schema``
            is empty.
        embed_dim: BART model dimension (``d_model``). Must be divisible by
            ``n_heads``. Default: 256.
        n_heads: Number of attention heads (encoder and decoder). Default: 8.
        n_layers: Number of encoder and decoder layers each. Default: 6.
        ffn_dim: Feed-forward dimension. Default: 4 * ``embed_dim``.
        prompt_length: Number of learnable soft-prompt positions prepended to
            the encoder. Default: 8.
        max_len: Maximum code-stream length (``max_position_embeddings``);
            streams are truncated to this length. Default: 512.
        mask_prob: Probability of masking each code token in the encoder input
            for the denoising objective. Default: 0.15.
        batch_size: Training batch size. Default: 16.
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
        >>> model = PromptEHR(
        ...     dataset, embed_dim=16, n_heads=2, n_layers=2, max_len=64
        ... )
        >>> isinstance(model, PromptEHR)
        True
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: Optional[int] = None,
        prompt_length: int = 8,
        max_len: int = 512,
        mask_prob: float = 0.15,
        batch_size: int = 16,
        epochs: int = 50,
        lr: float = 1e-4,
        save_dir: str = "./save/",
    ) -> None:
        super(PromptEHR, self).__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "PromptEHR expects an input feature named 'visits' backed by a "
                "NestedSequenceProcessor."
            )

        self.save_dir = save_dir
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.prompt_length = prompt_length

        # Code vocab from the NestedSequenceProcessor (includes <pad>=0, <unk>=1).
        self.visits_processor = dataset.input_processors["visits"]
        self.code_vocab_size = self.visits_processor.vocab_size()
        # Append five special tokens after the code vocab; reuse <pad>=0 as PAD.
        self.bos_id = self.code_vocab_size
        self.eos_id = self.code_vocab_size + 1
        self.delim_id = self.code_vocab_size + 2  # visit separator
        self.mask_id = self.code_vocab_size + 3  # denoising mask token
        self.code_prompt_id = self.code_vocab_size + 4  # modality prompt token
        self.pad_id = 0
        total_vocab_size = self.code_vocab_size + 5

        ffn_dim = ffn_dim if ffn_dim is not None else 4 * embed_dim
        config = BartConfig(
            vocab_size=total_vocab_size,
            max_position_embeddings=max_len,
            d_model=embed_dim,
            encoder_layers=n_layers,
            decoder_layers=n_layers,
            encoder_attention_heads=n_heads,
            decoder_attention_heads=n_heads,
            encoder_ffn_dim=ffn_dim,
            decoder_ffn_dim=ffn_dim,
            pad_token_id=self.pad_id,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            decoder_start_token_id=self.eos_id,  # BART convention
            forced_bos_token_id=None,
            forced_eos_token_id=None,
        )
        # Registered as sub-modules so .parameters()/.to() work.
        self.bart = BartForConditionalGeneration(config)
        # Learnable soft prompt (prompt learning), prepended to the encoder.
        self.soft_prompt = nn.Parameter(torch.zeros(prompt_length, embed_dim))
        nn.init.normal_(self.soft_prompt, std=config.init_std)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        """Resolve a user-supplied device, defaulting to CUDA when available."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Visit index tensor -> denoising seq2seq tensors
    # ------------------------------------------------------------------
    def _serialize(self, visits: torch.Tensor) -> List[List[int]]:
        """Serialize each patient's visits into a flat code stream.

        Layout (decoder target): ``[CODE_PROMPT] codes_v1 [VS] codes_v2 ...
        [EOS]``. Index 0 (``<pad>``) is skipped.
        """
        streams: List[List[int]] = []
        for i in range(visits.shape[0]):
            n_visits = int((visits[i].sum(dim=-1) > 0).sum().item())
            seq: List[int] = [self.code_prompt_id]
            for j in range(n_visits):
                codes = [int(c) for c in visits[i, j].tolist() if c > 0]
                seq.extend(codes)
                if j < n_visits - 1:
                    seq.append(self.delim_id)
            seq.append(self.eos_id)
            # Truncate but always keep the trailing EOS.
            if len(seq) > self.max_len:
                seq = seq[: self.max_len - 1] + [self.eos_id]
            streams.append(seq)
        return streams

    def _corrupt(self, stream: List[int]) -> List[int]:
        """Build the encoder input: BOS + a randomly masked copy + EOS.

        Code tokens are replaced with ``[MASK]`` with probability
        ``mask_prob``; the modality prompt and visit separators are preserved.
        This is the BART/PromptEHR span-infilling objective.
        """
        corrupted: List[int] = [self.bos_id]
        for tok in stream:
            if tok < self.code_vocab_size and torch.rand(1).item() < self.mask_prob:
                corrupted.append(self.mask_id)
            else:
                corrupted.append(tok)
        return corrupted

    def _encode_batch(self, visits: torch.Tensor):
        """Convert padded visit indices to encoder inputs and decoder labels.

        Returns:
            enc_input_ids: LongTensor ``(batch, L_enc)`` corrupted streams.
            enc_attention_mask: LongTensor ``(batch, L_enc)``.
            labels: LongTensor ``(batch, L_dec)`` full streams, padding -> -100.
        """
        streams = self._serialize(visits)
        enc_streams = [self._corrupt(s) for s in streams]

        enc_input_ids = self._pad_stack(enc_streams, self.pad_id)
        enc_attention_mask = (enc_input_ids != self.pad_id).long()
        # Position 0 is BOS, never masked out by the pad check; force it on.
        enc_attention_mask[:, 0] = 1

        labels = self._pad_stack(streams, self.pad_id)
        labels[labels == self.pad_id] = -100
        return enc_input_ids, enc_attention_mask, labels

    def _pad_stack(self, seqs: List[List[int]], pad_value: int) -> torch.Tensor:
        """Right-pad a list of int lists into a 2D LongTensor on ``self.device``."""
        length = max(len(s) for s in seqs)
        out = torch.full(
            (len(seqs), length), pad_value, dtype=torch.long, device=self.device
        )
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, device=self.device)
        return out

    def _encoder_inputs_embeds(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Prepend the learnable soft prompt to the encoder token embeddings.

        Returns the prompt-augmented ``inputs_embeds`` and the matching
        attention mask (soft-prompt positions are always attended to).
        """
        token_embeds = self.bart.get_input_embeddings()(input_ids)
        bsz = input_ids.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt, token_embeds], dim=1)
        prompt_mask = torch.ones(
            bsz, self.prompt_length, dtype=attention_mask.dtype, device=self.device
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        return inputs_embeds, attention_mask

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, visits: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass (denoising seq2seq reconstruction).

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)`` from
                the ``NestedSequenceProcessor``.
            **kwargs: Any other batch keys are ignored.

        Returns:
            Dict with ``loss`` (scalar seq2seq cross-entropy) and ``y_prob``
            (decoder next-token probabilities, shape ``(batch, L_dec, vocab)``).
        """
        visits = visits.to(self.device)
        enc_input_ids, enc_attention_mask, labels = self._encode_batch(visits)
        inputs_embeds, enc_attention_mask = self._encoder_inputs_embeds(
            enc_input_ids, enc_attention_mask
        )
        out = self.bart(
            inputs_embeds=inputs_embeds,
            attention_mask=enc_attention_mask,
            labels=labels,
        )
        return {"loss": out.loss, "y_prob": F.softmax(out.logits, dim=-1)}

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------
    def train_model(self, train_dataset, val_dataset=None, device=None) -> None:
        """Train PromptEHR with a custom loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Uses the standard ``get_dataloader``, an Adam
        optimizer, and the BART denoising loss. When ``val_dataset`` is given,
        validation loss is computed after each epoch and the best checkpoint is
        saved to ``self.save_dir``.

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)

        checkpoint_path = os.path.join(self.save_dir, "promptehr_model")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_loader = get_dataloader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        global_loss = 1e10
        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            self.bart.train()
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in batch_iter:
                visits = batch["visits"].to(self.device)

                optimizer.zero_grad()
                ret = self.forward(visits=visits)
                loss = ret["loss"]
                loss.backward()
                optimizer.step()
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            if val_dataset is not None:
                self.bart.eval()
                val_loader = get_dataloader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        visits = val_batch["visits"].to(self.device)
                        val_losses.append(self.forward(visits=visits)["loss"].item())

                cur_val_loss = float(np.mean(val_losses))
                print(f"Epoch {epoch} Validation Loss: {cur_val_loss:.7f}")
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        "model": self.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(state, checkpoint_path)
                    print("------------ Save best model ------------")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _decode_ids(self, ids: List[int], index_to_code: Dict[int, str]) -> List[List[str]]:
        """Decode a generated decoder token stream into per-visit code lists."""
        visits_out: List[List[str]] = []
        current: List[str] = []
        for tid in ids:
            if tid in (self.bos_id, self.pad_id, self.code_prompt_id, self.mask_id):
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
        """Generate synthetic patients with the trained PromptEHR model.

        Feeds the encoder a fully-masked seed stream (so generation is driven by
        the learned soft prompt), precomputes the prompt-augmented encoder
        states, and autoregressively samples a decoder stream with
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

        self.bart.eval()
        synthetic_dataset: List[Dict] = []
        sample_batch_size = min(num_samples, 256)
        generated = 0
        pbar = tqdm(total=num_samples, desc="Generating patients")

        # Fully-masked seed: [BOS] [CODE_PROMPT] [MASK] [EOS].
        seed = [self.bos_id, self.code_prompt_id, self.mask_id, self.eos_id]

        with torch.no_grad():
            while generated < num_samples:
                bs = min(sample_batch_size, num_samples - generated)
                enc_input_ids = torch.tensor(
                    [seed] * bs, dtype=torch.long, device=self.device
                )
                enc_attention_mask = torch.ones_like(enc_input_ids)
                inputs_embeds, enc_attention_mask = self._encoder_inputs_embeds(
                    enc_input_ids, enc_attention_mask
                )
                encoder_outputs = self.bart.get_encoder()(
                    inputs_embeds=inputs_embeds,
                    attention_mask=enc_attention_mask,
                    return_dict=True,
                )
                out_ids = self.bart.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=enc_attention_mask,
                    max_length=self.max_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=1,
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                    decoder_start_token_id=self.eos_id,
                )
                for i in range(bs):
                    # BART's generate prepends decoder_start_token_id (= eos_id)
                    # at position 0; skip it so the real eos terminates decoding.
                    visits_out = self._decode_ids(
                        out_ids[i].tolist()[1:], index_to_code
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
