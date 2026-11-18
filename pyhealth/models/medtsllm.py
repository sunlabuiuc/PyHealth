"""MedTsLLM-inspired ECG reprogramming model, ported to native PyHealth.

Contributors: Thi Van Trinh Tran and Zexi Li.
Paper: MedTsLLM, https://arxiv.org/abs/2408.07773.
This preserves the project's research reimplementation, not a verbatim port
of the authors' architecture. See the API documentation for differences.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel

PROMPT_MODES = ("none", "dataset", "task", "patient", "stats", "all")


class MedTsLLM(BaseModel):
    """Frozen GPT-2 or Llama-2 backbone with trainable ECG adaptation layers.

    Signals are [B, 512]. The output ``dense_logit`` is [B, 4, 512]. With a
    native packed time-series ``label``, ``logit``/``y_prob`` are [N_valid, 4]
    and ``y_true`` is the packed label tuple expected by PyHealth's Trainer.
    Ignored points are excluded, not reclassified as background.

    The research runner's sample normalization, zero-padded patches, attention,
    prompts, and linear interpolation are preserved. Prompt padding remains
    unmasked deliberately for compatibility; changing it needs a new analysis.
    The LLM is frozen but NOT run under no_grad: adapters need its input
    gradients. Full standard state_dict checkpoints include the backbone.
    Explicit compact checkpoints verify backbone identity before loading.

    Args:
        dataset: Native SampleDataset with ECGDelineationLUDB schemas.
        llm_name: Local Hugging Face snapshot path. Never downloaded by default.
        random_backbone: Initialize from config, then freeze. Seed beforehand.
        prompt_mode: none, dataset, task, patient, stats, or all.
        patch_len: Samples per patch, default 16.
        stride: Patch stride, default 8.
        d_model: Patch embedding dimension, default 64.
        num_tokens: Number of reprogramming prototypes, default 100.
        dropout: Adapter dropout, default 0.1.
        backbone: Optional injected backbone for offline synthetic examples/tests.
        tokenizer: Matching tokenizer; must be supplied with backbone.
        local_files_only: Prevent implicit model/tokenizer downloads by default.

    Examples:
        >>> model = MedTsLLM(  # doctest: +SKIP
        ...     dataset=train, llm_name="/models/gpt2-xl", prompt_mode="all",
        ... )
        >>> model.mode  # doctest: +SKIP
        'timeseries_multiclass'
    """

    def __init__(
        self, dataset, *, llm_name: str = "", random_backbone: bool = False,
        prompt_mode: str = "all", patch_len: int = 16, stride: int = 8,
        d_model: int = 64, num_tokens: int = 100, dropout: float = 0.1,
        backbone: nn.Module | None = None, tokenizer=None,
        local_files_only: bool = True,
    ) -> None:
        super().__init__(dataset)
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
        if any(not isinstance(x, int) or x <= 0
               for x in (patch_len, stride, d_model, num_tokens)):
            raise ValueError(
                "Patch/model/prototype dimensions must be positive integers"
            )
        if patch_len > 520 or not 0 <= dropout < 1:
            raise ValueError("patch_len must be <=520 and dropout in [0, 1)")
        if (backbone is None) != (tokenizer is None):
            raise ValueError("Supply backbone and tokenizer together")
        if dataset is not None and (
            self.label_processor.mode != "timeseries_multiclass"
        ):
            raise ValueError("MedTsLLM requires a timeseries_multiclass label")
        if dataset is not None and self.label_processor.output_size() != 4:
            raise ValueError("LUDB requires a fixed four-class output vocabulary")
        if dataset is not None and getattr(
            getattr(self.label_processor, "base", None), "label_vocab", None,
        ) != {0: 0, 1: 1, 2: 2, 3: 3}:
            raise ValueError(
                "LUDB label vocabulary must preserve class order [0,1,2,3]"
            )
        self.llm_name = str(llm_name)
        self.random_backbone = bool(random_backbone)
        self.prompt_mode = prompt_mode
        self.patch_len, self.stride = patch_len, stride
        self.window_length, self.num_classes = 512, 4
        self.n_patches = (520 - patch_len) // stride + 1
        self.adapter_config = {
            "patch_len": patch_len, "stride": stride, "d_model": d_model,
            "num_tokens": num_tokens, "dropout": dropout,
            "prompt_mode": prompt_mode, "random_backbone": self.random_backbone,
        }
        if backbone is None:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            config = AutoConfig.from_pretrained(
                llm_name, local_files_only=local_files_only,
            )
            config.output_hidden_states = False
            options = {}
            if config.model_type == "llama":
                config.use_cache = False
                options["torch_dtype"] = torch.float32
            self.llm = (
                AutoModel.from_config(config, **options) if random_backbone else
                AutoModel.from_pretrained(
                    llm_name, config=config, local_files_only=local_files_only,
                    **options,
                )
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_name, use_fast=True, local_files_only=local_files_only,
            )
        else:
            self.llm, self.tokenizer = backbone, tokenizer
        if self.llm.config.model_type not in {"gpt2", "llama"}:
            raise ValueError(
                "Only GPT-2-family and Llama-family backbones are supported"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token is None:
            raise ValueError("Tokenizer requires a pad token or an EOS token")
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.llm.eval()
        self.d_llm = int(self.llm.config.hidden_size)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.patch_to_llm = nn.Linear(d_model, self.d_llm)
        self.mapping_layer = nn.Linear(
            self.llm.get_input_embeddings().weight.shape[0], num_tokens,
        )
        heads = next(h for h in (8, 4, 2, 1) if self.d_llm % h == 0)
        self.reprogramming = nn.MultiheadAttention(
            self.d_llm, heads, dropout=dropout, batch_first=True,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, 4),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "word_embeddings",
            self.llm.get_input_embeddings().weight.detach().clone(),
            persistent=False,
        )

    def train(self, mode: bool = True):
        """Train adapters while keeping frozen backbone dropout disabled.

        Examples:
            >>> model.train().llm.training  # doctest: +SKIP
            False
        """
        super().train(mode)
        self.llm.eval()
        return self

    def _patches(self, signal: torch.Tensor) -> torch.Tensor:
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True).clamp_min(1e-5)
        padded = F.pad((signal - mean) / std, (0, 8))
        return self.patch_embedding(padded.unfold(1, self.patch_len, self.stride))

    def _prompt_text(
        self, signal: torch.Tensor, patient_context: Sequence[str] | None,
    ) -> list[str]:
        contexts = list(patient_context) if patient_context is not None else (
            [""] * signal.shape[0]
        )
        if len(contexts) != signal.shape[0] or any(
            not isinstance(context, str) for context in contexts
        ):
            raise ValueError("patient_context must have one string per sample")
        prompts = []
        for context, row in zip(contexts, signal.detach().float()):
            parts = []
            if self.prompt_mode in {"dataset", "all"}:
                parts.append("Dataset: LUDB 12-lead ECG, sampled at 500 Hz.")
            if self.prompt_mode in {"task", "all"}:
                parts.append(
                    "Task: label every ECG sample as background, P, QRS, or T."
                )
            if self.prompt_mode in {"patient", "all"} and context:
                parts.append(f"Patient information: {context}")
            if self.prompt_mode in {"stats", "all"}:
                parts.append(
                    f"Signal summary: mean={row.mean().item():.6g};"
                    f"std={row.std().item():.6g};min={row.min().item():.6g};"
                    f"max={row.max().item():.6g}"
                )
            prompts.append(" ".join(parts))
        return prompts

    def _dense_forward(self, signal: torch.Tensor, contexts) -> torch.Tensor:
        patches = self.patch_to_llm(self._patches(signal))
        prototypes = self.mapping_layer(self.word_embeddings.T).T
        key_value = prototypes.unsqueeze(0).expand(len(signal), -1, -1)
        attended, _ = self.reprogramming(
            patches, key_value, key_value, need_weights=False,
        )
        patches = self.dropout(attended + patches)
        texts = self._prompt_text(signal, contexts)
        if any(texts):
            tokens = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=False,
            ).input_ids.to(signal.device)
            prompts = self.llm.get_input_embeddings()(tokens)
        else:
            prompts = signal.new_empty((len(signal), 0, self.d_llm))
        hidden = torch.cat([prompts, patches], dim=1)
        if self.llm.config.model_type == "llama" and (
            hidden.shape[1] > self.llm.config.max_position_embeddings
        ):
            raise ValueError(
                "Prompt plus ECG patches exceeds Llama context; no truncation"
            )
        hidden = self.llm(inputs_embeds=hidden).last_hidden_state[:, -self.n_patches:]
        return F.interpolate(
            self.output_projection(hidden).transpose(1, 2), size=512,
            mode="linear", align_corners=False,
        )

    def forward(
        self, signal: torch.Tensor, label=None,
        ignore_mask: torch.Tensor | None = None,
        patient_context: Sequence[str] | None = None,
        dense_label: torch.Tensor | None = None, **kwargs,
    ) -> dict:
        """Return dense predictions and native ignore-aware packed predictions.

        All-ignored batches return an independent zero loss leaf, not a zero
        connected to model parameters. Thus Trainer's backward/step creates no
        parameter gradients or momentum updates (with gradients cleared to None).

        Examples:
            >>> output = model(**batch)  # doctest: +SKIP
            >>> output["dense_logit"].shape[-2:]  # doctest: +SKIP
            torch.Size([4, 512])
        """
        signal = signal.to(self.device)
        if signal.ndim == 3 and signal.shape[-1] == 1:
            signal = signal[..., 0]
        if signal.ndim != 2 or signal.shape[1] != 512 or len(signal) == 0:
            raise ValueError("Expected a non-empty signal batch [B, 512]")
        if signal.dtype != torch.float32 or not torch.isfinite(signal).all():
            raise ValueError("Research-compatible signals must be finite float32")
        logits = self._dense_forward(signal, patient_context)
        result = {"dense_logit": logits, "logit": logits,
                  "y_prob": logits.softmax(dim=1)}
        if label is None:
            return result
        if ignore_mask is None or ignore_mask.shape != signal.shape:
            raise ValueError("label requires an ignore_mask matching [B, 512]")
        if ignore_mask.dtype != torch.bool:
            raise ValueError("ignore_mask must be boolean")
        valid = ~ignore_mask.to(self.device)
        if not isinstance(label, (tuple, list)) or len(label) != 3:
            raise ValueError("label must be native (values, time, offsets)")
        values, time, offsets = [value.to(self.device) for value in label]
        if offsets.dtype != torch.long:
            raise ValueError("Packed label offsets must be integer indices")
        if offsets.ndim != 1 or len(offsets) != len(signal) + 1 or (
            offsets[0] != 0 or offsets[-1] != len(values)
        ):
            raise ValueError("Invalid packed label offsets")
        if values.ndim != 1 or time.ndim != 1 or len(time) != len(values):
            raise ValueError(
                "Expected matching one-dimensional packed values and times"
            )
        for index in range(len(signal)):
            expected = torch.nonzero(valid[index], as_tuple=True)[0]
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            if lo > hi or hi - lo != len(expected) or not torch.equal(
                time[lo:hi], expected.to(time.dtype),
            ):
                raise ValueError("Packed times disagree with the frozen ignore mask")
        if values.dtype != torch.long or (len(values) and (
            values.min() < 0 or values.max() > 3
        )):
            raise ValueError("Packed labels must be integer classes 0..3")
        if dense_label is not None:
            if dense_label.shape != signal.shape or not torch.equal(
                dense_label.to(self.device)[valid], values,
            ):
                raise ValueError("Packed values disagree with dense frozen labels")
        packed = logits.permute(0, 2, 1)[valid]
        result.update({
            "logit": packed, "y_prob": packed.softmax(dim=-1),
            "y_true": (values, time, offsets),
            "loss": F.cross_entropy(packed, values) if len(values) else torch.zeros(
                (), device=self.device, requires_grad=torch.is_grad_enabled(),
            ),
            "supervised_points": len(values),
        })
        return result

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a normal full checkpoint, refreshing nonpersistent prototypes.

        Examples:
            >>> model.load_state_dict(model.state_dict())  # doctest: +SKIP
        """
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.word_embeddings = self.llm.get_input_embeddings().weight.detach().clone()
        return result

    def _backbone_fingerprint(self) -> str:
        digest = hashlib.sha256()
        config = self.llm.config.to_dict()
        for key in ("_name_or_path", "transformers_version"):
            config.pop(key, None)
        digest.update(json.dumps(config, sort_keys=True, default=str).encode())
        for name, value in self.llm.state_dict().items():
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            flat = value.detach().reshape(-1)
            for chunk in flat.split(1_000_000):
                digest.update(
                    chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
        return digest.hexdigest()

    def save_adapters(
        self, path: str | Path, *, provenance: dict | None = None,
    ) -> None:
        """Save compact adapters with a content-verified backbone reference.

        Does not overwrite an existing file. Fingerprinting reads all backbone
        weights; compact files are not self-contained, especially for random
        frozen backbones. Preserve the exact backbone separately.

        Examples:
            >>> model.save_adapters("final_adapters.pt")  # doctest: +SKIP
        """
        payload = {
            "format": "pyhealth-medtsllm-adapters-v1",
            "adapter_config": self.adapter_config,
            "backbone_fingerprint": self._backbone_fingerprint(),
            "backbone_location": self.llm_name,
            "provenance": provenance or {},
            "model_state": {key: value.detach().cpu() for key, value in
                            self.state_dict().items() if not key.startswith("llm.")},
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    def load_adapters(self, path: str | Path) -> dict:
        """Load only compatible adapters; reject unknown backbone provenance.

        Historical adapter-only checkpoints lack this fingerprint and are not
        silently accepted. Full standard checkpoints still use load_state_dict.

        Examples:
            >>> provenance = model.load_adapters("final_adapters.pt")  # doctest: +SKIP
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "pyhealth-medtsllm-adapters-v1":
            raise ValueError("Unverified or unsupported compact checkpoint format")
        if payload["adapter_config"] != self.adapter_config:
            raise ValueError("Adapter configuration mismatch")
        if payload["backbone_fingerprint"] != self._backbone_fingerprint():
            raise ValueError("Frozen backbone fingerprint mismatch")
        expected = {key: value for key, value in self.state_dict().items()
                    if not key.startswith("llm.")}
        incoming = payload["model_state"]
        if incoming.keys() != expected.keys() or any(
            incoming[key].shape != expected[key].shape for key in expected
        ):
            raise ValueError("Adapter keys or tensor shapes do not match")
        super().load_state_dict(incoming, strict=False)
        return payload["provenance"]
