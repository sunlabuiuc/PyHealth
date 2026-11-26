pyhealth.interpret.methods.attingrad
====================================

Overview
--------

AttInGrad gives token-level explanations for medical coding models with
class-wise attention, such as :class:`~pyhealth.models.PLMCA`. For a code
``j`` and token ``n`` it multiplies the decoder attention ``A[j, n]`` by
InputXGrad, the L2 norm over the embedding dimension of the input embedding
times the gradient of the code's logit:

.. math::

    \mathrm{AttInGrad}_{j,n} = A_{j,n} \cdot
    \left\lVert X_n \odot \frac{\partial f_j}{\partial X_n} \right\rVert_2

Attention alone is often misaligned with the input words, because after many
encoder layers every position mixes the whole chunk. InputXGrad is measured at
the input, so the product keeps only the tokens both point to. The method is
from Edin et al. (2024), "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records" (https://arxiv.org/abs/2406.08958).

The same interpreter also returns its parts and an ablation variant through
``method``:

- ``"attention"``: the decoder attention only
- ``"inputxgrad"``: InputXGrad only
- ``"attingrad"``: attention x InputXGrad (default)
- ``"integrated_gradients"``: Integrated Gradients on the embeddings from a
  mask-token baseline (special tokens kept), reduced per token with the L2 norm
- ``"att_ig"``: attention x Integrated Gradients
- ``"deeplift"``: DeepLIFT (Rescale rule) on the embeddings from the same
  mask-token baseline, reduced per token with the L2 norm; it reuses the hook
  context of :class:`~pyhealth.interpret.methods.DeepLift`
- ``"att_deeplift"``: attention x DeepLIFT

Every explanation is L1-normalized so one note's token scores sum to 1;
padding gets 0.

Usage Notes
-----------

1. **Model compatibility**: the model needs ``tokenize``,
   ``get_input_embeddings`` and ``forward_from_embedding`` returning
   ``logit`` and ``attention``; :class:`~pyhealth.models.PLMCA` has all three.
2. **Tokens**: scores have shape ``[batch, N]`` and line up with
   ``model.tokenize(texts)``, including the special tokens.
3. **Target code**: pass ``target_class_idx``; without it each note's
   highest-scoring code is explained.
4. **Evaluation**: compare explanations with human evidence spans using
   :func:`pyhealth.metrics.interpretability.plausibility_metrics`.

Quick Start
-----------

.. code-block:: python

    from pyhealth.interpret.methods import AttInGrad

    # model is a trained PLMCA, batch a dataloader batch with a "text" key
    interpreter = AttInGrad(model, method="attingrad")
    scores = interpreter.attribute(**batch, target_class_idx=3)["text"]
    input_ids, _ = model.tokenize(batch["text"])
    top = scores[0].topk(5).indices
    print(model.tokenizer.convert_ids_to_tokens(input_ids[0, top]))

See ``examples/mdace_icd9_coding_plm_ca_attingrad.py`` for training PLM-CA on
MIMIC-III and evaluating every method on MDACE evidence spans.

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.AttInGrad
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
