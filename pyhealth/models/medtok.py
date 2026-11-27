"""MedTok: multimodal medical code tokenizer embeddings for PyHealth models.

Paper: Xiaorui Su et al. "Multimodal Medical Code Tokenizer." ICML 2025.
https://arxiv.org/abs/2502.04397
Official code (MIT): https://github.com/mims-harvard/MedTok
Released tokenizer and embeddings (MIT): https://huggingface.co/mims-harvard/MedTok

MedTok encodes each medical code from its text description (a language model)
and its place in a biomedical knowledge graph (a graph encoder), and quantizes
both into one shared codebook of 21,000 tokens; every code becomes 20 tokens.
The authors release, for about 616,000 codes (ICD-9, ICD-10-CM, ICD-10-PCS,
ATC, SNOMED CT, RxNorm, ...), the 20 token ids of each code
(``code2tokens.json``) and its 256-dimensional quantized embedding
(``code2embeddings.json``). Training the tokenizer itself is not reproduced
here: it needs PrimeKG and four GPUs for three days.

This module uses the released files as a drop-in replacement for the code
embedding table of any PyHealth model built on
:class:`~pyhealth.models.EmbeddingModel` (Transformer, RNN, RETAIN, ...), as
the paper swaps the tokenizer of existing EHR models:

* :class:`MedTokEmbedding` with ``mode="embedding"`` looks each code up in the
  released 256-dimensional embeddings and projects them to the model's
  ``embedding_dim``, with one learned vector for codes MedTok does not cover,
  as the official EHR code does.
* ``mode="tokens"`` embeds each code as the mean of learned embeddings of its
  20 MedTok tokens, so codes that share tokens share parameters.
* :func:`apply_medtok` swaps these into a built model, one feature at a time.

MedTok's keys are dotted codes with no vocabulary prefix (``"428.0"``,
``"E11.9"``, ``"A10BA02"``), and plain numeric codes from other vocabularies
share the key space, so an undotted MIMIC ICD-9 code such as ``"4280"`` must
be converted first; :func:`medtok_keys` does this for each code system.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

MEDTOK_DIM = 256
MEDTOK_TOKENS = 21000
CODE_SYSTEMS = (
    "ICD9CM",
    "ICD9PROC",
    "ICD10CM",
    "ICD10PCS",
    "ICD_DX",
    "ICD_PROC",
    "ATC",
    "NDC",
    "RAW",
)
# ICD_DX / ICD_PROC codes carry their version, as MIMIC-IV tasks write them.
_VERSIONED = {
    "ICD_DX": {"9": "ICD9CM", "10": "ICD10CM"},
    "ICD_PROC": {"9": "ICD9PROC", "10": "ICD10PCS"},
}
_SPECIAL = ("<pad>", "<unk>")
_NDC_TO_ATC = None


def _ndc_to_atc(code: str) -> list[str]:
    global _NDC_TO_ATC
    if _NDC_TO_ATC is None:
        from pyhealth.medcode import CrossMap

        _NDC_TO_ATC = CrossMap.load("NDC", "ATC")
    return list(_NDC_TO_ATC.map(code))


def medtok_keys(code: Any, system: str) -> list[str]:
    """Converts a dataset code to the MedTok keys to look up.

    Args:
        code: The code as the dataset stores it (MIMIC stores ICD codes
            without the dot, e.g. ``"4280"``).
        system: One of ``"ICD9CM"``, ``"ICD9PROC"``, ``"ICD10CM"``,
            ``"ICD10PCS"``, ``"ICD_DX"`` / ``"ICD_PROC"`` (diagnosis or
            procedure codes prefixed with their ICD version, ``"9_4280"`` or
            ``"10_E119"``, as the MIMIC-IV readmission, length-of-stay and
            drug-recommendation tasks write them), ``"ATC"``, ``"NDC"``
            (mapped to its ATC codes with :class:`~pyhealth.medcode.CrossMap`)
            or ``"RAW"`` (used as is).

    Returns:
        The candidate keys; empty for a blank code and for PyHealth's
        ``<pad>`` and ``<unk>`` tokens. An NDC can map to several ATC codes,
        whose embeddings are then averaged.

    Raises:
        ValueError: For an unknown ``system``, or an ``ICD_DX`` / ``ICD_PROC``
            code without a ``9_`` or ``10_`` prefix. Unprefixed MIMIC-IV codes
            mix ICD-9 and ICD-10 and cannot be told apart (``E8889`` is both
            ICD-9 E888.9 and ICD-10 E88.89).

    Examples:
        >>> from pyhealth.models.medtok import medtok_keys
        >>> medtok_keys("4280", "ICD9CM"), medtok_keys("E8889", "ICD9CM")
        (['428.0'], ['E888.9'])
        >>> medtok_keys("3893", "ICD9PROC"), medtok_keys("E119", "ICD10CM")
        (['38.93'], ['E11.9'])
        >>> medtok_keys("9_E8889", "ICD_DX"), medtok_keys("10_E8889", "ICD_DX")
        (['E888.9'], ['E88.89'])
    """
    if system not in CODE_SYSTEMS:
        raise ValueError(f"system must be one of {CODE_SYSTEMS}, got {system!r}.")
    c = str(code).strip()
    if c in _SPECIAL:
        return []
    if system in _VERSIONED:
        version, _, rest = c.partition("_")
        if version not in _VERSIONED[system] or not rest:
            raise ValueError(
                f"{system} codes look like '9_4280' or '10_E119', got {code!r}."
            )
        return medtok_keys(rest, _VERSIONED[system][version])
    if ":" in c and system != "RAW":
        c = c.split(":", 1)[1].strip()  # e.g. "ICD9CM:4280"
    if not c:
        return []
    if system in ("ICD9CM", "ICD9PROC", "ICD10CM"):
        c = c.upper().replace(".", "")
        split = (
            2
            if system == "ICD9PROC"
            else 4
            if c.startswith("E") and system == "ICD9CM"
            else 3
        )
        return [c[:split] + "." + c[split:] if len(c) > split else c]
    if system in ("ICD10PCS", "ATC"):
        return [c.upper()]
    if system == "NDC":
        return _ndc_to_atc(c)
    return [c]


_WS = re.compile(r"\s*")
# An error this close to the buffer's end may be a value cut by the chunk
# (inside a number, a \uXXXX escape, NaN or Infinity): read more and retry.
_CUT_MARGIN = 12


class _NeedMore(Exception):
    pass


def _decode(decoder: json.JSONDecoder, buf: str, pos: int) -> tuple[Any, int]:
    """``raw_decode`` that tells a value cut by the buffer's end from a bad one."""
    try:
        return decoder.raw_decode(buf, pos)
    except json.JSONDecodeError as e:
        if e.msg.startswith("Unterminated string") or e.pos >= len(buf) - _CUT_MARGIN:
            raise _NeedMore from None
        raise ValueError(f"Malformed MedTok JSON: {e}") from None


def iter_medtok_json(
    path: str, keys: Iterable[str] | None = None, chunk_size: int = 1 << 22
) -> Iterator[tuple[str, list]]:
    """Streams ``(code, value)`` pairs from a MedTok ``code2*.json`` file.

    The released ``code2embeddings.json`` is 4 GB, too large for
    ``json.load`` on most laptops, so it is decoded one entry at a time.

    Args:
        path: ``code2embeddings.json`` or ``code2tokens.json`` (a JSON object
            from code to a list of numbers).
        keys: Only yield these codes. Default None yields all.
        chunk_size: Characters read at a time. Default 4 Mi.

    Yields:
        ``(code, list of numbers)``.

    Raises:
        ValueError: If the file is not a JSON object of code to list (for
            example a missing comma or colon), or is empty or truncated.

    Examples:
        >>> import json, tempfile, os
        >>> from pyhealth.models.medtok import iter_medtok_json
        >>> path = os.path.join(tempfile.mkdtemp(), "code2tokens.json")
        >>> _ = open(path, "w").write(json.dumps({"428.0": [3, 7], "I10": [1, 2]}))
        >>> list(iter_medtok_json(path, keys={"I10"}))
        [('I10', [1, 2])]
    """
    wanted = None if keys is None else set(keys)
    decoder = json.JSONDecoder()
    # What may come next: "{" (start), "first" (an entry or "}"), "entry"
    # (after a comma) or "sep" (a comma or "}").
    expect = "{"
    with open(path, encoding="utf-8") as f:
        buf, pos, eof = "", 0, False
        while True:
            try:
                cur = _WS.match(buf, pos).end()
                if cur >= len(buf):
                    raise _NeedMore
                char = buf[cur]
                if expect == "{":
                    if char != "{":
                        raise ValueError(f"{path} is not a JSON object.")
                    pos, expect = cur + 1, "first"
                    continue
                if expect == "sep" or (expect == "first" and char == "}"):
                    if char == "}":
                        rest = buf[cur + 1 :] + f.read()
                        if rest.strip():
                            raise ValueError(f"{path} has data after the object.")
                        return
                    if char != ",":
                        raise ValueError(f"{path}: expected ',' or '}}' at {cur}.")
                    pos, expect = cur + 1, "entry"
                    continue
                # An entry: "code" : [values]. Nothing moves until it is whole.
                if char != '"':
                    raise ValueError(f"{path}: expected a quoted code at {cur}.")
                code, cur = _decode(decoder, buf, cur)
                cur = _WS.match(buf, cur).end()
                if cur >= len(buf):
                    raise _NeedMore
                if buf[cur] != ":":
                    raise ValueError(f"{path}: expected ':' after {code!r}.")
                cur = _WS.match(buf, cur + 1).end()
                value, cur = _decode(decoder, buf, cur)
                if not isinstance(value, list):  # bad file content, not a bad argument
                    raise ValueError(f"{path}: {code!r} is not a list.")  # noqa: TRY004
            except _NeedMore:
                if eof:
                    raise ValueError(f"{path} is empty or truncated.") from None
                more = f.read(chunk_size)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            pos, expect = cur, "sep"
            if wanted is None or code in wanted:
                yield code, value


def convert_medtok_embeddings(json_path: str, out_path: str) -> int:
    """Converts ``code2embeddings.json`` once into a compact ``.npz``.

    The ``.npz`` (about 630 MB, float32) loads in seconds; the JSON takes
    minutes to parse. Pass either file to :func:`load_medtok`.

    Args:
        json_path: The released ``code2embeddings.json``.
        out_path: Output ``.npz`` path, with arrays ``codes`` and
            ``embeddings``.

    Returns:
        The number of codes written.

    Examples:
        >>> import json, tempfile, os
        >>> from pyhealth.models.medtok import convert_medtok_embeddings
        >>> tmp = tempfile.mkdtemp()
        >>> src = os.path.join(tmp, "code2embeddings.json")
        >>> _ = open(src, "w").write(json.dumps({"I10": [0.5] * 256}, indent=2))
        >>> convert_medtok_embeddings(src, os.path.join(tmp, "medtok.npz"))
        1
    """
    codes, rows = [], []
    for code, vec in iter_medtok_json(json_path):
        if len(vec) != MEDTOK_DIM:
            raise ValueError(f"{code!r} has {len(vec)} values, expected {MEDTOK_DIM}.")
        codes.append(code)
        rows.append(np.asarray(vec, dtype=np.float32))
    np.savez(out_path, codes=np.asarray(codes), embeddings=np.stack(rows))
    return len(codes)


def load_medtok(path: str, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Loads MedTok embeddings or token ids for the given keys.

    Args:
        path: ``code2embeddings.json``, ``code2tokens.json`` or an ``.npz``
            from :func:`convert_medtok_embeddings`.
        keys: Only load these MedTok keys (see :func:`medtok_keys`). Default
            None loads all.

    Returns:
        Dict from MedTok key to a 1D array: 256 floats for embeddings, 20 (or
        a multiple of 20) token ids for tokens.

    Examples:
        >>> import json, tempfile, os
        >>> from pyhealth.models.medtok import load_medtok
        >>> path = os.path.join(tempfile.mkdtemp(), "code2tokens.json")
        >>> _ = open(path, "w").write(json.dumps({"428.0": [3, 7], "I10": [1, 2]}))
        >>> load_medtok(path, keys=["428.0", "missing"])["428.0"].tolist()
        [3, 7]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MedTok file not found: {path}")
    wanted = None if keys is None else set(keys)
    if path.endswith(".npz"):
        archive = np.load(path)
        codes, table = archive["codes"], archive["embeddings"]
        return {
            str(c): table[i]
            for i, c in enumerate(codes)
            if wanted is None or str(c) in wanted
        }
    return {code: np.asarray(v) for code, v in iter_medtok_json(path, wanted)}


class MedTokEmbedding(nn.Module):
    """Embeds a feature's codes with MedTok, as a drop-in for ``nn.Embedding``.

    Indices are those of a PyHealth processor's ``code_vocab``. Padding
    (``<pad>``) embeds to zeros, so PyHealth models still mask it; codes
    MedTok does not cover, and ``<unk>``, share one learned vector.

    It is not an ``nn.Embedding`` subclass, since it has no single weight
    table. Code that only looks for ``nn.Embedding`` layers in a model, such
    as ``pyhealth.autoimprove``'s code explanations and
    :class:`~pyhealth.models.DescEmb`'s pretrained-weight transfer, skips
    the features MedTok replaced.

    Args:
        code_vocab: The processor's code-to-index mapping.
        medtok: :func:`load_medtok` output (embeddings for
            ``mode="embedding"``, token ids for ``mode="tokens"``), or its
            path.
        system: The feature's code system (see :func:`medtok_keys`).
        embedding_dim: Output size (the model's ``embedding_dim``).
        mode: ``"embedding"`` (released 256-d vectors, then a linear
            projection) or ``"tokens"`` (mean of learned token embeddings).
            A code with several MedTok keys (an NDC mapped to several ATC
            codes) gets the mean of their vectors, or the mean over all of
            their tokens. Default ``"embedding"``.
        freeze: In ``"embedding"`` mode, keep the MedTok vectors fixed (the
            projection and missing-code vector still train), as the official
            EHR code does. Default True.

    Attributes:
        coverage: Fraction of the vocabulary's real codes (not ``<pad>`` or
            ``<unk>``) found in MedTok.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.models.medtok import MedTokEmbedding
        >>> vocab = {"<pad>": 0, "<unk>": 1, "4280": 2, "9999": 3}
        >>> emb = MedTokEmbedding(
        ...     vocab, {"428.0": np.ones(256)}, "ICD9CM", embedding_dim=16
        ... )
        >>> emb.coverage
        0.5
        >>> out = emb(torch.tensor([[2, 3, 0]]))
        >>> out.shape, bool(out[0, 2].abs().sum() == 0)
        (torch.Size([1, 3, 16]), True)
    """

    def __init__(
        self,
        code_vocab: Mapping[Any, int],
        medtok: Mapping[str, np.ndarray] | str,
        system: str,
        embedding_dim: int,
        mode: str = "embedding",
        freeze: bool = True,
    ):
        super().__init__()
        if mode not in ("embedding", "tokens"):
            raise ValueError(f"mode must be 'embedding' or 'tokens', got {mode!r}.")
        self.mode = mode
        self.system = system
        self.embedding_dim = embedding_dim
        self.padding_idx = code_vocab.get("<pad>", 0)
        size = max(code_vocab.values()) + 1
        keys = {
            tok: medtok_keys(tok, system)
            for tok in code_vocab
            if str(tok) not in _SPECIAL
        }
        if isinstance(medtok, str):
            medtok = load_medtok(medtok, {k for ks in keys.values() for k in ks})

        known = torch.zeros(size, dtype=torch.bool)
        table = torch.zeros(size, MEDTOK_DIM)
        token_rows = {}  # tokens mode: vocab index -> all of its codes' token ids
        for tok, ks in keys.items():
            found = {k: np.asarray(medtok[k]) for k in ks if k in medtok}
            if not found:
                continue
            for k, v in found.items():
                if mode == "embedding" and v.shape != (MEDTOK_DIM,):
                    raise ValueError(
                        f"MedTok embedding of {k!r} has shape {v.shape}, expected "
                        f"({MEDTOK_DIM},); for code2tokens.json use mode='tokens'."
                    )
                if mode == "tokens" and (
                    v.ndim != 1
                    or v.dtype.kind not in "iu"
                    or v.min(initial=0) < 0
                    or v.max(initial=0) >= MEDTOK_TOKENS
                ):
                    raise ValueError(
                        f"MedTok tokens of {k!r} are not ids in [0, {MEDTOK_TOKENS}); "
                        "mode='tokens' needs code2tokens.json."
                    )
            vals = list(found.values())
            idx = code_vocab[tok]
            known[idx] = True
            if mode == "embedding":
                table[idx] = torch.as_tensor(np.mean(vals, axis=0), dtype=torch.float32)
            else:
                token_rows[idx] = torch.as_tensor(
                    np.concatenate(vals), dtype=torch.long
                )
        if mode == "tokens":
            # Sized after concatenation, so an NDC keeps every ATC code's tokens.
            width = max((len(ids) for ids in token_rows.values()), default=1)
            table = torch.full((size, width), -1, dtype=torch.long)
            for idx, ids in token_rows.items():
                table[idx, : len(ids)] = ids
        self.coverage = float(known.sum()) / max(len(keys), 1)
        self.register_buffer("known", known)

        if mode == "embedding":
            if freeze:
                self.register_buffer("table", table)
            else:
                self.table = nn.Parameter(table)
            self.missing = nn.Parameter(torch.randn(MEDTOK_DIM) * 0.02)
            self.proj = nn.Linear(MEDTOK_DIM, embedding_dim)
        else:
            self.register_buffer("token_ids", table)
            self.tokens = nn.Embedding(MEDTOK_TOKENS, embedding_dim)
            self.missing = nn.Parameter(torch.randn(embedding_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds ``x`` (code indices, any shape) to ``x.shape + (embedding_dim,)``."""
        x = x.long()
        known = self.known[x].unsqueeze(-1)
        if self.mode == "embedding":
            vec = torch.where(known, self.table[x], self.missing)
            out = self.proj(vec)
        else:
            ids = self.token_ids[x]  # (..., 20) with -1 padding
            valid = (ids >= 0).unsqueeze(-1)
            summed = (self.tokens(ids.clamp(min=0)) * valid).sum(-2)
            vec = summed / valid.sum(-2).clamp(min=1)
            out = torch.where(known, vec, self.missing)
        return out.masked_fill((x == self.padding_idx).unsqueeze(-1), 0.0)


def apply_medtok(
    model: nn.Module,
    medtok: Mapping[str, np.ndarray] | str,
    code_systems: Mapping[str, str],
    mode: str = "embedding",
    freeze: bool = True,
) -> dict[str, float]:
    """Replaces a PyHealth model's code embeddings with :class:`MedTokEmbedding`.

    Works with any model whose codes are embedded by
    :class:`~pyhealth.models.EmbeddingModel` (``model.embedding_model``),
    such as :class:`~pyhealth.models.Transformer`,
    :class:`~pyhealth.models.RNN` and :class:`~pyhealth.models.RETAIN`.
    Call it before creating the optimizer.

    Args:
        model: A built PyHealth model.
        medtok: :func:`load_medtok` output or path (embeddings or tokens,
            matching ``mode``). A path is read once for all features.
        code_systems: Feature key to code system, e.g.
            ``{"conditions": "ICD9CM", "procedures": "ICD9PROC"}``. Features
            not listed keep their embedding.
        mode: See :class:`MedTokEmbedding`.
        freeze: See :class:`MedTokEmbedding`.

    Returns:
        Feature key to its MedTok coverage.

    Raises:
        ValueError: If the model has no ``embedding_model`` or a listed
            feature has no code vocabulary.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.models.medtok import apply_medtok
        >>> samples = [
        ...     {"patient_id": "p0", "conditions": ["4280", "4019"], "label": 1},
        ...     {"patient_id": "p1", "conditions": ["4280"], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"conditions": "sequence"}, {"label": "binary"}
        ... )
        >>> model = Transformer(dataset, embedding_dim=16)
        >>> apply_medtok(model, {"428.0": np.ones(256)}, {"conditions": "ICD9CM"})
        {'conditions': 0.5}
    """
    embedding_model = getattr(model, "embedding_model", None)
    if embedding_model is None:
        raise ValueError(f"{type(model).__name__} has no embedding_model to replace.")
    processors = embedding_model.dataset.input_processors
    vocabs = {}
    for feature in code_systems:
        vocab = getattr(processors.get(feature), "code_vocab", None)
        if vocab is None or feature not in embedding_model.embedding_layers:
            raise ValueError(f"Feature {feature!r} has no code embedding to replace.")
        vocabs[feature] = vocab
    if isinstance(medtok, str):
        wanted = {
            k
            for f, vocab in vocabs.items()
            for tok in vocab
            if str(tok) not in _SPECIAL
            for k in medtok_keys(tok, code_systems[f])
        }
        medtok = load_medtok(medtok, wanted)

    coverage = {}
    for feature, vocab in vocabs.items():
        old = embedding_model.embedding_layers[feature]
        # The old layer may be an nn.Embedding or, on a second call, a
        # MedTokEmbedding; both have embedding_dim.
        device = next(iter([*old.parameters(), *old.buffers()])).device
        layer = MedTokEmbedding(
            vocab,
            medtok,
            code_systems[feature],
            embedding_dim=old.embedding_dim,
            mode=mode,
            freeze=freeze,
        ).to(device)
        embedding_model.embedding_layers[feature] = layer
        coverage[feature] = layer.coverage
    return coverage
