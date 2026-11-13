pyhealth.interpret.methods.token_masking
========================================

Overview
--------

Token masking (TM) is a training objective that makes a model's explanations
more plausible by teaching it to ignore irrelevant words. It fine-tunes an
already trained model (the student) next to a frozen copy of it (the
teacher). For each batch:

1. :func:`~pyhealth.interpret.methods.optimize_token_mask` learns a soft mask
   ``M`` over the tokens with AdamW (up to ``max_iter`` steps, stopping when
   the loss changes by less than 3% over two steps). It minimises

   .. math::

       \frac{1}{BN}\lVert \sigma(M) \rVert_1 + \beta \frac{1}{B}
       \lVert f_s(X) - f_s(X \odot \sigma(M) + E_b \odot (1 - \sigma(M)))
       \rVert_1

   on predicted probabilities, where ``E_b`` is the embedding of the
   baseline token. The model's own gradients are not touched.

2. The mask is rounded, special tokens and padding are always kept, and
   :func:`~pyhealth.interpret.methods.token_masking_loss` returns

   .. math::

       \frac{1}{B}\lVert f_s(X) - f_t(X) \rVert_1 + \lambda \frac{1}{B}
       \lVert f_s(X) - f_s(X_{\text{masked}}) \rVert_1

   with the first ``f_s(X)`` of the second term detached.

The method is from Edin et al. (2024), "An Unsupervised Approach to Achieve
Supervised-Level Explainability in Healthcare Records"
(https://arxiv.org/abs/2406.08958), Section 2.2 and Appendix D, following the
authors' ``masking_loss`` and ``token_masking`` code. Their TM run fine-tunes
a trained model for one epoch with learning rate 1e-5, batch size 16,
``lambda_1 = 1``, ``diet_constant = 0.01`` and mask learning rate 0.1.

Usage Notes
-----------

1. **Model compatibility**: the model needs ``tokenizer``,
   ``get_input_embeddings`` and ``forward_from_embedding`` returning
   ``logit``; :class:`~pyhealth.models.PLMCA` and
   :class:`~pyhealth.models.CAML` have all three.
2. **Teacher**: pass the frozen starting model's logits on the same batch as
   ``teacher_logits``.
3. **Baseline token**: defaults to the tokenizer's mask token. The authors'
   TM runs used id 50001, an unused row of the RoBERTa-PM embedding.

Quick Start
-----------

.. code-block:: python

    import copy
    import torch
    from pyhealth.interpret.methods import token_masking_loss

    # model is a trained PLMCA or CAML, batch a dataloader batch
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    input_ids, mask = model.tokenize(batch["text"])
    with torch.no_grad():
        embeds = teacher.get_input_embeddings()(input_ids)
        teacher_logits = teacher.forward_from_embedding(embeds, mask)["logit"]
    out = token_masking_loss(model, input_ids, mask, teacher_logits)
    out["loss"].backward()
    optimizer.step()

See ``examples/mdace_icd9_coding_plm_ca_attingrad.py`` (``--tm``) for TM
fine-tuning on MIMIC-III and MDACE.

API Reference
-------------

.. autofunction:: pyhealth.interpret.methods.optimize_token_mask

.. autofunction:: pyhealth.interpret.methods.token_masking_loss
