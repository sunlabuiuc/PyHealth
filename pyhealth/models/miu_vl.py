"""MIU-VL prompts: attribute descriptions of medical targets for GLIP.

Paper: Qin et al., "Medical Image Understanding with Pretrained Vision
Language Models: A Comprehensive Study", ICLR 2023,
https://arxiv.org/abs/2209.15517. Code: https://github.com/MembrAI/MIU-VL.

A grounding detector such as :class:`~pyhealth.models.GLIP` finds common
objects from their names but not medical targets: the prompt ``"polyp"``
gets 6.1 AP on CVC-300. MIU-VL describes the target instead, by its colour,
shape and location. A prompt is a phrase dict, as in MIU-VL's prompt files:
``name`` is the scored text, ``prefix`` and ``suffix`` are context.

- Manual prompts are written by hand; :data:`POLYP_MANUAL_PROMPT` is the
  polyp prompt of MIU-VL's configuration.
- MLM prompts (:func:`mlm_prompts`) ask PubMedBERT for each attribute with a
  cloze sentence, e.g. ``"The typical color of polyp is [MASK]."``, and fill
  the k-th ranked words into MIU-VL's template
  ``"{color} color, {shape} shape " + name + " in {location}"``.

MIU-VL's released script passes the segment ids to the masked language
model as its second positional argument, which is the attention mask, so it
masks every token and predicts punctuation (``"."``, ``"-"``, ``"to"``).
:func:`mlm_attributes` calls the model with the attention mask, which gives
the words of the paper's examples (``pink``, ``oval``, ``rectum``).
"""

from __future__ import annotations

from typing import Any

PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
"""The PubMedBERT checkpoint used for MLM prompts (formerly ``BiomedNLP-PubMedBERT-...``)."""

PUBMEDBERT_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"

MLM_TEMPLATES = {
    "color": "The typical color of {} is [MASK].",
    "shape": "The typical shape of {} is [MASK].",
    "location": "The location of {} is at [MASK].",
}
"""MIU-VL's cloze sentences for each attribute (``make_autopromptsv2.py``)."""

POLYP_MANUAL_PROMPT = {"prefix": "", "name": "pink yellow red color oval bump", "suffix": " in rectum"}
"""MIU-VL's manual polyp prompt (``configs/glip_Swin_T_O365_GoldG_polyp_kvasir.yaml``)."""


def mlm_attributes(
    class_name: str,
    topk: int = 3,
    model_name: str = PUBMEDBERT,
    revision: str | None = PUBMEDBERT_REVISION,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, list[str]]:
    """The ``topk`` words a masked language model predicts per attribute.

    Args:
        class_name: The target's name in the cloze sentences, e.g. ``"polyp"``.
        topk: Words per attribute, best first.
        model_name: Hugging Face masked language model.
        revision: Model revision (pinned for PubMedBERT).
        model: A loaded ``AutoModelForMaskedLM``, instead of ``model_name``.
        tokenizer: Its tokenizer, required with ``model``.

    Returns:
        ``{"color": [...], "shape": [...], "location": [...]}``.

    Examples:
        >>> mlm_attributes("polyp")  # doctest: +SKIP
        {'color': ['yellow', 'red', 'pink'], 'shape': ['oval', 'round', 'irregular'],
         'location': ['rectum', 'right', 'fundus']}
    """
    import torch

    if model is None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
    elif tokenizer is None:
        raise ValueError("tokenizer is required with model.")
    model.eval()
    words = {}
    for attribute, template in MLM_TEMPLATES.items():
        enc = tokenizer(template.format(class_name), return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        position = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
        with torch.no_grad():
            logits = model(**enc).logits[0, position]
        words[attribute] = tokenizer.convert_ids_to_tokens(logits.topk(topk).indices.tolist())
    return words


def mlm_prompts(
    class_name: str,
    object_name: str | None = None,
    topk: int = 3,
    attributes: dict[str, list[str]] | None = None,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """MIU-VL's MLM prompts: the k-th prompt uses each attribute's k-th word.

    Args:
        class_name: The target's name in the cloze sentences.
        object_name: The scored word of the prompt; MIU-VL uses ``"bump"``
            for polyps. Defaults to ``class_name``.
        topk: Number of prompts.
        attributes: Precomputed :func:`mlm_attributes`; computed if omitted.
        **kwargs: Passed to :func:`mlm_attributes`.

    Returns:
        ``topk`` phrase dicts, best first.

    Examples:
        >>> words = {"color": ["pink"], "shape": ["oval"], "location": ["rectum"]}
        >>> mlm_prompts("polyp", "bump", topk=1, attributes=words)
        [{'prefix': 'pink color, oval shape ', 'name': 'bump', 'suffix': ' in rectum'}]
    """
    words = attributes or mlm_attributes(class_name, topk=topk, **kwargs)
    name = object_name or class_name
    return [
        {
            "prefix": f"{words['color'][k]} color, {words['shape'][k]} shape ",
            "name": name,
            "suffix": f" in {words['location'][k]}",
        }
        for k in range(topk)
    ]
