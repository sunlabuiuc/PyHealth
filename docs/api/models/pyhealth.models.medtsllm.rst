pyhealth.models.medtsllm
======================

Native ``BaseModel`` port of the project's MedTsLLM-inspired LUDB model.
Contributors: Thi Van Trinh Tran and Zexi Li.
Reference: `MedTsLLM <https://arxiv.org/abs/2408.07773>`_.

Implementation boundary
-----------------------

This preserves the research implementation's per-window mean/std scaling,
zero-padded patches, PyTorch multihead reprogramming attention, residual/dropout,
layernorm/linear patch head, and interpolated 512-point output. It is **not**
a verbatim reproduction of the authors' RevIN/custom attention/flattening-head
implementation. Full-window ECG segmentation is not a causal forecasting task.

The frozen GPT-2 XL or Llama-2-7B base backbone remains in eval mode while
gradients flow through it into trainable adapters. No LoRA, LLM-weight tuning,
quantization, mixed precision, prompt truncation, or new mask policy is added.
The original prompt token padding has no attention mask; this known behavior
is retained for research compatibility, not recommended as a new design.

Prompt modes are none, dataset only, task only, patient only, statistics only,
and all. The nine-condition research matrix also compares correct/mismatched/
removed patient context and pretrained/random frozen backbones. Each condition
uses seeds 0, 1 and 2. The combined all-correct pretrained condition is counted
once, yielding 27 runs per backbone, 54 for GPT-2 XL plus Llama-2-7B.

The example ``examples/ludb_segmentation_medtsllm.py`` supports that matrix,
one supplied backbone at a time. Real runs require explicit source paths, a
local snapshot, CUDA and a new output directory. The default research budget
is 10 epochs, batch size 32, AdamW lr=1e-4, weight decay=0, clip norm=1, FP32.
The fixed supplied patient split is preserved; there is no CV, early stopping,
test-based selection, or implicit validation split. Report the final epoch.

Context mismatch rotates the sorted **entire metadata cohort**, including
across the train/test boundary, matching the historical runner. Removal clears
the entire context string. Both interventions operate during training AND
testing. They are not the proposal's separate fixed-checkpoint, test-time-only
context intervention; that experiment remains distinct and is not claimed here.

Native Trainer compatibility
----------------------------

``get_dataloader`` uses the task's native processors. The model returns packed
valid-position logits/probabilities and labels for ordinary Trainer metrics,
plus ``dense_logit`` for the independently defined ECG metrics. Generic Trainer
metrics are not the custom segment mIoU and may handle absent classes
differently; use the ECG helper for the documented fixed-class summaries.

.. code-block:: python

    import torch
    from pyhealth.datasets import get_dataloader
    from pyhealth.models import MedTsLLM
    from pyhealth.trainer import Trainer

    model = MedTsLLM(train, llm_name="/models/gpt2-xl/snapshot")
    trainer = Trainer(model, device="cuda", enable_logging=False)
    trainer.train(
        get_dataloader(train, batch_size=32, shuffle=True, seed=0),
        epochs=10, optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 1e-4}, weight_decay=0.0, max_grad_norm=1.0,
        load_best_model_at_last=False,
    )

This Trainer example demonstrates API compatibility; its native dataloader
shuffle is not the historical GPU permutation. For matching that batching
policy, use the explicit research loop in the example script. Both approaches
exclude ignored points from the loss. An all-ignored batch's independent zero
loss creates no parameter gradients when gradients are cleared to None; the research
loop additionally skips its optimizer step explicitly.

Checkpoint safety
-----------------

Normal ``state_dict``/``load_state_dict`` retain standard PyTorch behavior and
include the backbone, so standard Trainer checkpoints can be large. Loading
refreshes the nonpersistent word-prototype buffer. ``save_adapters`` and
``load_adapters`` are explicit compact alternatives: they check adapter
configuration, tensor keys/shapes, and a SHA-256 fingerprint of backbone
configuration/weights. Fingerprinting costs a complete backbone read.
Compact save refuses to overwrite existing files.

Pretrained compact checkpoints require the original local backbone snapshot.
Random frozen backbones need the exact random weights, not just the name of
a pretrained snapshot; the example also saves a full checkpoint for random
conditions. Preserve the tokenizer, software versions, seed and input
provenance. Fingerprints verify backbone weights, not arbitrary tokenizer
edits. Historical adapter-only checkpoints lack this format's fingerprint and
are deliberately rejected by ``load_adapters``; no silent partial loading.

Validation status
-----------------

Synthetic tests are supplied for dataset/task APIs, loss/mask behavior,
gradients, Trainer use and checkpoint round trips. They have not yet been
executed as part of preparing this patch. Current PyHealth dependencies differ
from the historical Biowulf environment; runtime compatibility and numerical
parity remain unverified. Nothing here implies rerunning completed experiments
or replacing their results. No model weights, tokens or patient data are
included in the contribution.

.. autoclass:: pyhealth.models.medtsllm.MedTsLLM
    :members:
    :show-inheritance:
