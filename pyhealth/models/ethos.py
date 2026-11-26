"""ETHOS: a generative model of patient timelines, with zero-shot risk estimates.

ETHOS is a GPT-style decoder trained to predict the next token of a tokenized
patient timeline. It needs no labels: to estimate a risk it samples many future
timelines from the prediction time and counts how often the outcome happens.

Paper:
    Renc et al. Foundation Model of Electronic Medical Records for Adaptive
    Risk Estimation. https://arxiv.org/abs/2502.06124

Adapted from ethos-ares (commit 2d54383), ``src/ethos/model.py`` and
``src/ethos/inference/{inference,utils}.py``:
https://github.com/ipolharvard/ethos-ares

MIT License. Copyright (c) 2024 Paweł Renc.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import EthosContextProcessor, EthosProcessor
from pyhealth.processors.ethos_processor import TIMELINE_END_TOKEN
from pyhealth.tasks.ethos_tasks import (
    INVALID_TOKEN,
    STOP_TOKEN,
    TIME_LIMIT,
    TOKEN_LIMIT,
    EthosRolloutRule,
)

from .base_model import BaseModel

HOURS_PER_YEAR = 24 * 365.25
IGNORE_INDEX = -100


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention without bias terms."""

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.size()
        q, k, v = (
            part.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for part in self.c_attn(x).split(width, dim=2)
        )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(y))


class _MLP(nn.Module):
    """The feed-forward part of a transformer block."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.activation = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.activation(self.c_fc(x))))


class _Block(nn.Module):
    """A pre-norm transformer block."""

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = _CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = _MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


def _age_quantiles(
    age_years: torch.Tensor, num_quantiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Writes ages as two digits in base ``num_quantiles``, as ETHOS does.

    With 10 quantiles, age 46 is ``(4, 6)``: the tokens ``Q5`` and ``Q7``.

    Args:
        age_years: Ages in years.
        num_quantiles: Number of quantile tokens.

    Returns:
        Both digits, from 0, as ``long`` tensors. ``nan`` gives ``-1``.
    """
    scaled = (age_years * num_quantiles**2 / 100).clamp(0, num_quantiles**2 - 1)
    high = torch.div(scaled, num_quantiles, rounding_mode="floor")
    low = torch.round(scaled - high * num_quantiles)
    carry = low == num_quantiles
    high, low = high + carry, low.masked_fill(carry, 0)
    missing = age_years.isnan()
    return (
        high.nan_to_num(0).long().masked_fill(missing, -1),
        low.nan_to_num(0).long().masked_fill(missing, -1),
    )


class ETHOS(BaseModel):
    """ETHOS generative timeline model with zero-shot risk estimation.

    :meth:`forward` is the next-token loss, so the model trains with
    :class:`~pyhealth.trainer.Trainer` on
    :class:`~pyhealth.tasks.EthosTimelineMEDS`, with no labels. Its dataset has
    no label processor, so ``mode`` is ``None`` and there is no ``y_prob``. Risks come from :meth:`estimate_risk`,
    which samples timelines forward from each sample of
    :class:`~pyhealth.tasks.EthosOutcomeMEDS`.

    The input is the static context followed by the timeline, as in ETHOS. A
    timeline that does not fit in ``n_positions`` gives a random window in
    training and its last tokens otherwise, and the age tokens state the age
    where that window starts. Demographics keep their value at the start of the
    whole timeline.

    The defaults are the paper's final model. Layer names match ``ethos-ares``.
    Reload a checkpoint only with the fitted processors it was trained on: the
    token ids mean nothing without them.

    Args:
        dataset: A dataset with an ``"ethos"`` field named ``timeline`` and an
            ``"ethos_context"`` field named ``context``.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_embd: Embedding size.
        n_positions: Context size, including the static context.
        dropout: Dropout rate.

    Raises:
        ValueError: If ``dataset`` lacks those fields or has a label, or the
            sizes do not fit.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import ETHOS
        >>> from pyhealth.tasks import EthosRolloutRule
        >>> start = datetime(2024, 1, 1)
        >>> codes = ["HOSPITAL_ADMISSION//ER", "LAB//A", "MEDS_DEATH"]
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "timeline": {
        ...             "time": [start + timedelta(hours=h) for h in (0, 1, 2)],
        ...             "code": codes,
        ...             "numeric_value": [None, 1.0 + i, None],
        ...         },
        ...         "context": {
        ...             "time": [datetime(1980, 1, 1)],
        ...             "code": ["MEDS_BIRTH"],
        ...             "start": start,
        ...         },
        ...     }
        ...     for i in range(3)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"timeline": "ethos", "context": "ethos_context"},
        ...     output_schema={},
        ... )
        >>> model = ETHOS(dataset, n_layer=1, n_head=2, n_embd=8, n_positions=32)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=3)))
        >>> model(**batch)["loss"].backward()
        >>> rule = EthosRolloutRule(("MEDS_DEATH", "TIMELINE_END"), ("MEDS_DEATH",))
        >>> risk = model.estimate_risk(
        ...     [batch], rule, num_rollouts=4, max_new_tokens=8, seed=0
        ... )
        >>> risk["y_prob"].shape
        torch.Size([3])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_layer: int = 6,
        n_head: int = 12,
        n_embd: int = 768,
        n_positions: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(dataset=dataset)
        if self.label_processors:
            raise ValueError(
                "ETHOS trains without labels, so its dataset needs an empty "
                f"output_schema, not {sorted(self.label_processors)}; see "
                "EthosTimelineMEDS. Score labels with estimate_risk."
            )

        processors = dataset.input_processors
        timeline = processors.get("timeline")
        context = processors.get("context")
        has_fields = isinstance(timeline, EthosProcessor) and isinstance(
            context, EthosContextProcessor
        )
        if not has_fields:
            raise ValueError(
                "ETHOS needs a 'timeline' field of type 'ethos' and a 'context' "
                "field of type 'ethos_context'; see EthosTimelineMEDS."
            )
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} must divide by n_head {n_head}.")
        if n_positions <= context.context_size:
            raise ValueError(
                f"n_positions {n_positions} leaves no room after the "
                f"{context.context_size} context tokens."
            )
        self.timeline_processor = timeline
        self.context_processor = context
        self.n_positions = n_positions
        self.context_size = context.context_size
        self.timeline_size = n_positions - context.context_size

        # Context tokens sit after the timeline's. The vocabulary is rounded up
        # to a multiple of 64, as in ETHOS, so it can have ids that are not
        # tokens; sampling one is an invalid token.
        self.timeline_vocab_size = timeline.size()
        self.vocab_size = math.ceil((timeline.size() + context.size()) / 64) * 64

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.vocab_size, n_embd),
                "wpe": nn.Embedding(n_positions, n_embd),
                "drop": nn.Dropout(dropout),
                "h": nn.ModuleList(
                    [_Block(n_embd, n_head, dropout) for _ in range(n_layer)]
                ),
                "ln_f": nn.LayerNorm(n_embd, bias=False),
            }
        )
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        vocab = timeline.vocab
        self.register_buffer(
            "quantile_ids",
            torch.tensor(
                [vocab[f"Q{k + 1}"] for k in range(timeline.num_quantiles)],
                dtype=torch.long,
            ),
            persistent=False,
        )
        # On the CPU in float64: not every device has that type.
        self._interval_hours = torch.zeros(self.vocab_size, dtype=torch.float64)
        for label, hours in timeline.interval_estimates.items():
            self._interval_hours[vocab[label]] = hours
        self._is_token = torch.zeros(self.vocab_size, dtype=torch.bool)
        self._is_token[: self.timeline_vocab_size] = True
        self._is_token[[vocab.pad_id, vocab.unk_id]] = False
        self._id_to_token = {i: token for token, i in vocab.token_to_id.items()}

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Returns next-token logits for token ids of shape ``[B, L]``."""
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.transformer["wte"](ids) + self.transformer["wpe"](positions)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        return self.lm_head(self.transformer["ln_f"](x))

    def _context_ids(self, static: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """Builds model ids of the static context.

        Args:
            static: Output tokens of the context processor, ``[B, C]``.
            age: Age in years where the timeline window starts, ``[B]``.

        Returns:
            Context ids in the model's vocabulary, ``[B, C]``.
        """
        ids = static + self.timeline_vocab_size
        unknown = self.timeline_processor.vocab.unk_id
        position = self.context_processor.age_position
        digits = _age_quantiles(age, self.timeline_processor.num_quantiles)
        for offset, digit in enumerate(digits):
            tokens = self.quantile_ids[digit.clamp(min=0)]
            ids[:, position + offset] = tokens.masked_fill(digit < 0, unknown)
        return ids

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Computes the next-token loss.

        Args:
            **kwargs: A batch, with ``timeline`` and ``context``.

        Returns:
            A dictionary with ``loss``, ``logit`` of shape ``[B, L, V]`` and
            ``y_true``, the target ids, with ``-100`` where there is none.
        """
        time, tokens = kwargs["timeline"]
        age, static = kwargs["context"]
        tokens = tokens.to(self.device)
        time = time.to(self.device, dtype=torch.float32)
        static = static.to(self.device)
        age = age.to(self.device, dtype=torch.float32).view(-1)
        if tokens.size(1) == 0:
            tokens = tokens.new_zeros(len(tokens), 1)
            time = time.new_zeros(len(time), 1)

        pad_id = self.timeline_processor.vocab.pad_id
        lengths = (tokens != pad_id).sum(dim=1)
        spare = (lengths - self.timeline_size - 1).clamp(min=0)
        if self.training:
            starts = (torch.rand_like(spare, dtype=torch.float32) * (spare + 1)).long()
        else:
            starts = spare
        width = min(tokens.size(1), self.timeline_size + 1)
        indices = starts[:, None] + torch.arange(width, device=self.device)
        window = tokens.gather(1, indices.clamp(max=tokens.size(1) - 1))
        window = window.masked_fill(indices >= lengths[:, None], pad_id)

        age = age + time.gather(1, starts[:, None]).squeeze(1) / HOURS_PER_YEAR
        context = self._context_ids(static, age)
        ids = torch.cat([context, window[:, :-1]], dim=1)
        labels = torch.cat(
            [
                torch.full_like(context, IGNORE_INDEX),
                window[:, 1:].masked_fill(window[:, 1:] == pad_id, IGNORE_INDEX),
            ],
            dim=1,
        )

        logits = self._logits(ids)
        if (labels != IGNORE_INDEX).any():
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        else:
            # No timeline has a next token. A zero that keeps the graph, instead
            # of the NaN mean of nothing.
            loss = logits.sum() * 0.0
        return {"loss": loss, "logit": logits, "y_true": labels}

    def _token_mask(self, prefixes: Iterable[str]) -> torch.Tensor:
        """Marks the vocabulary's tokens that start with one of ``prefixes``."""
        prefixes = tuple(prefixes)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for index, token in self._id_to_token.items():
            mask[index] = token.startswith(prefixes)
        return mask

    def _prompt(
        self,
        time: torch.Tensor,
        tokens: torch.Tensor,
        age: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Builds the ids that a rollout starts from, ``[1, L]``.

        Args:
            time: One sample's hours per token, ``[L]``.
            tokens: One sample's tokens, ``[L]``, possibly padded.
            age: The sample's age in years.
            static: The sample's context tokens, ``[C]``.

        Returns:
            The context, then the last tokens that fit. A closing
            ``TIMELINE_END`` is left out, since the timeline continues here.
        """
        vocab = self.timeline_processor.vocab
        kept = tokens != vocab.pad_id
        time, tokens = time[kept], tokens[kept].to(self.device)
        if len(tokens) and tokens[-1] == vocab[TIMELINE_END_TOKEN]:
            time, tokens = time[:-1], tokens[:-1]
        start = max(len(tokens) - self.timeline_size, 0)
        elapsed = time[start].item() if len(time) else 0.0
        age_at_start = age.to(self.device, dtype=torch.float32).view(-1)
        age_at_start = age_at_start + elapsed / HOURS_PER_YEAR
        context = self._context_ids(static.to(self.device).view(1, -1), age_at_start)
        return torch.cat([context, tokens[start:].view(1, -1)], dim=1)

    @torch.no_grad()
    def rollout(
        self,
        timeline: tuple[torch.Tensor, torch.Tensor],
        context: tuple[torch.Tensor, torch.Tensor],
        rule: EthosRolloutRule,
        num_rollouts: int = 100,
        temperature: float = 1.0,
        max_new_tokens: int = 8192,
        generator: torch.Generator | None = None,
    ) -> dict[str, list]:
        """Samples future timelines for one sample.

        Each rollout samples one token at a time, adding up the mean duration
        of each interval token, until ``rule`` stops it, it passes the time
        limit, it samples an id that is not a timeline token, or it reaches
        ``max_new_tokens``.

        Args:
            timeline: ``(time, tokens)`` of one sample, both ``[L]``.
            context: ``(age, tokens)`` of one sample.
            rule: When to stop.
            num_rollouts: Number of timelines to sample.
            temperature: Softmax temperature.
            max_new_tokens: Limit on generated tokens per rollout. ETHOS has
                none, but a poorly trained model may never advance the clock.
            generator: Source of randomness, on this model's device.

        Returns:
            A dictionary of lists, one element per rollout: ``tokens`` (the
            generated ids), ``actual`` (the last token, or ``None`` if invalid),
            ``reason`` (why it stopped) and ``days`` (simulated days).
        """
        was_training = self.training
        self.eval()
        prompt = self._prompt(*timeline, *context)
        ids = prompt.repeat(num_rollouts, 1)
        stops = self._token_mask(rule.stop_prefixes)
        limit = rule.time_limit_days * 24.0

        alive = torch.arange(num_rollouts)
        hours = torch.zeros(num_rollouts, dtype=torch.float64)
        generated: list[list[int]] = [[] for _ in range(num_rollouts)]
        reasons = [TOKEN_LIMIT] * num_rollouts
        for _ in range(max_new_tokens):
            logits = self._logits(ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=generator)

            if ids.size(1) == self.n_positions:
                ids = torch.cat(
                    [ids[:, : self.context_size], ids[:, self.context_size + 1 :]],
                    dim=1,
                )
            ids = torch.cat([ids, next_ids], dim=1)

            drawn = next_ids.view(-1).cpu()
            hours[alive] += self._interval_hours[drawn]
            for index, token in zip(alive.tolist(), drawn.tolist()):
                generated[index].append(token)

            invalid = ~self._is_token[drawn]
            late = hours[alive] > limit
            done = invalid | late | stops[drawn]
            for row in done.nonzero().view(-1).tolist():
                reason = STOP_TOKEN
                if late[row]:
                    reason = TIME_LIMIT
                if invalid[row]:
                    reason = INVALID_TOKEN
                reasons[alive[row]] = reason

            alive, ids = alive[~done], ids[~done.to(ids.device)]
            if len(alive) == 0:
                break

        self.train(was_training)
        return {
            "tokens": generated,
            "actual": [
                self._id_to_token[tokens[-1]]
                if tokens and self._is_token[tokens[-1]]
                else None
                for tokens in generated
            ],
            "reason": reasons,
            "days": (hours / 24.0).tolist(),
        }

    def _check_processors(self, dataset: Any) -> None:
        """Raises if ``dataset`` encodes with other processors than the model's."""
        processors = getattr(dataset, "input_processors", None)
        if not processors:
            return
        ours = {"timeline": self.timeline_processor, "context": self.context_processor}
        for key, mine in ours.items():
            theirs = processors.get(key)
            if theirs is mine:
                continue
            same = (
                theirs is not None
                and type(theirs) is type(mine)
                and theirs.vocab is not None
                and theirs.vocab.to_json() == mine.vocab.to_json()
                and getattr(theirs, "quantiles", None) == getattr(mine, "quantiles", None)
                and getattr(theirs, "interval_estimates", None)
                == getattr(mine, "interval_estimates", None)
            )
            if not same:
                raise ValueError(
                    f"The dataset's {key!r} processor is not the one this model "
                    "was trained with, so its token ids mean something else. "
                    "Build the outcome dataset with "
                    "input_processors=model.dataset.input_processors."
                )

    def _check_ids(self, tokens: torch.Tensor, static: torch.Tensor) -> None:
        """Raises if a batch has ids outside the model's vocabularies."""
        if tokens.numel() and int(tokens.max()) >= self.timeline_vocab_size:
            raise ValueError(
                "A timeline token id is outside this model's vocabulary; the "
                "batch was encoded with other processors. Build the outcome "
                "dataset with input_processors=model.dataset.input_processors."
            )
        if static.numel() and int(static.max()) >= self.context_processor.size():
            raise ValueError(
                "A context token id is outside this model's vocabulary; build "
                "the outcome dataset with "
                "input_processors=model.dataset.input_processors."
            )

    def estimate_risk(
        self,
        dataloader: Iterable[dict[str, Any]],
        rule: EthosRolloutRule,
        num_rollouts: int = 100,
        temperature: float = 1.0,
        max_new_tokens: int = 8192,
        label_key: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Estimates each sample's risk from sampled timelines.

        The risk is the share of positive rollouts among those ``rule`` keeps.
        The paper uses 100 rollouts per sample.

        The batches must be encoded with this model's processors: build the
        outcome dataset with ``input_processors=model.dataset.input_processors``
        (see :class:`~pyhealth.tasks.EthosOutcomeMEDS`). Processors fitted on
        the outcome samples give token ids that mean something else, so this
        raises ``ValueError`` when the dataloader's dataset has other
        processors, or when a token id falls outside the model's vocabulary.

        Rollouts have no key-value cache: each new token re-runs the model over
        the whole sequence so far. That is fine for small models and short
        rollouts, but it will be slow at the paper's scale (2048-token context,
        100 rollouts per sample).

        Args:
            dataloader: Batches with ``timeline`` and ``context``.
            rule: How rollouts stop and count, such as
                :attr:`pyhealth.tasks.EthosOutcomeMEDS.rule`.
            num_rollouts: Rollouts per sample.
            temperature: Softmax temperature.
            max_new_tokens: Limit on generated tokens per rollout.
            label_key: A label to return as ``y_true``.
            seed: Seed for reproducible rollouts.

        Returns:
            A dictionary with ``y_prob`` (``nan`` if every rollout was dropped),
            ``num_kept`` (rollouts behind each risk), ``patient_id`` (if the
            batches have it) and ``y_true`` (with ``label_key``).
        """
        self._check_processors(getattr(dataloader, "dataset", None))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        risks: list[float] = []
        kept_counts: list[int] = []
        patient_ids: list[Any] = []
        labels: list[torch.Tensor] = []
        for batch in dataloader:
            time, tokens = batch["timeline"]
            age, static = batch["context"]
            self._check_ids(tokens, static)
            for i in range(len(tokens)):
                result = self.rollout(
                    (time[i], tokens[i]),
                    (age[i], static[i]),
                    rule,
                    num_rollouts=num_rollouts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    generator=generator,
                )
                scores = [
                    rule.score(actual, reason, days)
                    for actual, reason, days in zip(
                        result["actual"], result["reason"], result["days"]
                    )
                ]
                kept = [score for score in scores if score is not None]
                risks.append(sum(kept) / len(kept) if kept else math.nan)
                kept_counts.append(len(kept))
            patient_ids.extend(batch.get("patient_id", []))
            if label_key is not None:
                labels.append(batch[label_key].view(-1).cpu())

        output: dict[str, Any] = {
            "y_prob": torch.tensor(risks, dtype=torch.float32),
            "num_kept": torch.tensor(kept_counts, dtype=torch.long),
        }
        if patient_ids:
            output["patient_id"] = patient_ids
        if label_key is not None:
            output["y_true"] = torch.cat(labels)
        return output
