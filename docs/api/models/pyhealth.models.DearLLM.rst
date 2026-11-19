pyhealth.models.DearLLM
=======================

Overview
--------

DearLLM (Xu et al., `DearLLM: Enhancing Personalized Healthcare via Large
Language Models-Deduced Feature Correlations
<https://doi.org/10.1609/aaai.v39i1.32079>`_, AAAI 2025) adds a per-patient
graph of LLM-deduced correlations between diagnosis codes to an EHR sequence
model.

1. **Feature prior graph**: the nodes are the patient's diagnosis codes. The
   weight of the edge ``c_i -> c_j`` comes from the conditional perplexity of
   a medical LLM on the name of ``c_j`` after a prompt with the patient's
   visit history and "``c_i`` is related to". Low perplexity means a strong
   correlation. The graph is built once, offline, and is an input to the
   model.
2. **Graph encoder**: code embeddings followed by GCN layers over the
   weighted, directed graph (:class:`~pyhealth.models.DearLLMGraphEncoder`,
   plain PyTorch, no graph library needed).
3. **Feature-frequency-aware pooling**: node vectors are summed with weights
   ``softmax(CF-IPF)``, a TF-IDF style score that favours codes frequent for
   this patient but rare in the training set.
4. **Predictor**: the graph vector is concatenated with the output of a GRU,
   LSTM or Transformer over the visits and fed to a linear layer.

Inputs
------

Besides the EHR code features, the input schema needs three graph fields:
``graph_nodes`` (``"sequence"``), ``graph_node_weights`` (``"tensor"``) and
``graph_edges`` (``"tensor"`` of ``[source, target, weight]`` rows).
:class:`~pyhealth.tasks.DearLLMMortalityPredictionMIMIC3` and
:class:`~pyhealth.tasks.DearLLMMortalityPredictionMIMIC4` produce them; see
``examples/mortality_prediction/dearllm_mimic3.py`` for the full pipeline,
including the optional LLM step.

.. note::

   Differences from the paper and the official code: HiTANet and StageNet
   backbones are not included (GRU, LSTM and Transformer are); the GCN has
   no self-loops by default, as in the official code (the paper's Eq. 4 adds
   them; set ``add_self_loops=True``); CF-IPF follows the paper's Eq. 5-6
   rather than scikit-learn's ``TfidfVectorizer`` used in the official code.

API
---

.. autoclass:: pyhealth.models.DearLLM
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DearLLMGraphEncoder
    :members:
    :undoc-members:
    :show-inheritance:
