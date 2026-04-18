Models
===============

PyHealth models sit between the :doc:`processors` (which turn raw patient data
into tensors) and the :doc:`trainer` (which runs the training loop). Each
model takes a ``SampleDataset`` — the result of ``dataset.set_task()`` — as
its first constructor argument, and uses it to automatically build the right
embedding layers and output head for your task.

One thing worth knowing up front: the ``SampleDataset`` carries fitted
processor metadata that the model needs to configure itself. If you pass the
raw ``BaseDataset`` instead you'll get an error, because it hasn't been
processed into samples yet.

Choosing a Model
----------------

The table below covers the most commonly used models and when each one fits
best. If your features are a mix of sequential codes and static numeric
vectors, ``MultimodalRNN`` is usually the easiest starting point because it
routes each feature type automatically.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Good fit when…
     - Notes
   * - :doc:`models/pyhealth.models.RNN`
     - Your features are sequences of medical codes (diagnoses, procedures, drugs) across visits
     - One RNN per feature, hidden states concatenated; ``rnn_type`` can be ``"GRU"`` (default) or ``"LSTM"``
   * - :doc:`models/pyhealth.models.Transformer`
     - You have longer code histories and want attention to capture long-range dependencies
     - Self-attention across the sequence; tends to work well when visit order matters
   * - :doc:`models/pyhealth.models.MLP`
     - Features are static numeric vectors (aggregated lab values, demographics)
     - Fully connected; no notion of sequence order
   * - ``MultimodalRNN``
     - Features mix sequential codes with static tensors or multi-hot encodings
     - Auto-routes sequential features to RNN layers and non-sequential features to linear layers; good default for EHR
   * - :doc:`models/pyhealth.models.StageNet`
     - You have time-stamped vital signs with irregular measurement intervals
     - Requires ``StageNetProcessor`` or ``StageNetTensorProcessor`` in the task schema
   * - :doc:`models/pyhealth.models.GNN`
     - Features include graph-structured data
     - Works with ``GraphProcessor``; see :doc:`graph` for setup
   * - :doc:`models/pyhealth.models.GraphCare`
     - You want to augment EHR codes with a medical knowledge graph
     - Combines code sequences with a :class:`~pyhealth.graph.KnowledgeGraph`

How BaseModel Works
--------------------

All PyHealth models inherit from ``BaseModel``, which itself inherits from
PyTorch's ``nn.Module``. When you call ``MyModel(dataset=sample_ds)``, the
base class reads the dataset's schemas and automatically sets:

- ``self.feature_keys`` — the list of input field names from ``input_schema``
- ``self.label_keys`` — the list of output field names from ``output_schema``
- ``self.device`` — the compute device

It also provides three helper methods that take care of the boilerplate that
varies by task type:

- ``self.get_output_size()`` returns the output dimension from the fitted
  label processor, so you don't have to hard-code it.
- ``self.get_loss_function()`` returns the right loss for the task: BCE for
  binary and multilabel tasks, cross-entropy for multiclass, MSE for
  regression.
- ``self.prepare_y_prob(logits)`` applies sigmoid, softmax, or identity to
  logits depending on the task, producing calibrated probabilities.

The ``forward()`` method is expected to return a dictionary with four keys:
``loss``, ``y_prob``, ``y_true``, and ``logit``. The Trainer reads all four.

EmbeddingModel
--------------

:class:`~pyhealth.models.EmbeddingModel` is a helper that routes each input
feature to the appropriate embedding layer based on how its processor works.
Features from token-based processors (``SequenceProcessor``,
``NestedSequenceProcessor``, and similar) get a learned ``nn.Embedding``
lookup. Features from continuous processors (``TensorProcessor``,
``TimeseriesProcessor``, ``MultiHotProcessor``) get a linear projection
instead. You end up with a uniform embedding shape across all features:

.. code-block:: python

    self.embedding_model = EmbeddingModel(dataset, embedding_dim=128)
    embedded = self.embedding_model(inputs, masks=masks)
    # embedded[key] has shape (batch_size, seq_len, embedding_dim)

Task Mode and Loss Functions
-----------------------------

PyHealth automatically selects the loss function and output activation based
on the label processor in your task's ``output_schema``:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Output schema value
     - Loss function
     - ``y_prob`` shape and activation
   * - ``"binary"``
     - BCE with logits
     - sigmoid → (batch, 1)
   * - ``"multiclass"``
     - Cross-entropy
     - softmax → (batch, num_classes)
   * - ``"multilabel"``
     - BCE with logits
     - sigmoid → (batch, num_labels)
   * - ``"regression"``
     - MSE
     - identity → (batch, 1)

Building a Custom Model
-----------------------

If none of the built-in models fit your architecture, you can subclass
``BaseModel`` directly. The skeleton below shows the typical structure: build
an ``EmbeddingModel`` in ``__init__``, unpack processor schemas in
``forward``, pool or aggregate the embeddings, and return the four-key dict.

.. code-block:: python

    from pyhealth.models import BaseModel
    from pyhealth.models.embedding import EmbeddingModel
    import torch
    import torch.nn as nn

    class MyModel(BaseModel):
        def __init__(self, dataset, embedding_dim=128):
            super().__init__(dataset=dataset)
            self.label_key = self.label_keys[0]
            self.embedding_model = EmbeddingModel(dataset, embedding_dim)
            self.fc = nn.Linear(embedding_dim * len(self.feature_keys),
                                self.get_output_size())

        def forward(self, **kwargs):
            inputs, masks = {}, {}
            for key in self.feature_keys:
                feature = kwargs[key]
                if isinstance(feature, torch.Tensor):
                    feature = (feature,)
                schema = self.dataset.input_processors[key].schema()
                inputs[key] = feature[schema.index("value")]
                if "mask" in schema:
                    masks[key] = feature[schema.index("mask")]

            embedded = self.embedding_model(inputs, masks=masks)
            pooled = [embedded[k].mean(dim=1) for k in self.feature_keys]
            logits = self.fc(torch.cat(pooled, dim=1))

            y_true = kwargs[self.label_key].to(self.device)
            return {
                "loss":   self.get_loss_function()(logits, y_true),
                "y_prob": self.prepare_y_prob(logits),
                "y_true": y_true,
                "logit":  logits,
            }

API Reference
-------------

.. toctree::
    :maxdepth: 3

    models/pyhealth.models.BaseModel
    models/pyhealth.models.LogisticRegression
    models/pyhealth.models.MLP
    models/pyhealth.models.CNN
    models/pyhealth.models.RNN
    models/pyhealth.models.GNN
    models/pyhealth.models.Transformer
    models/pyhealth.models.TransformersModel
    models/pyhealth.models.RETAIN
    models/pyhealth.models.GAMENet
    models/pyhealth.models.GraphCare
    models/pyhealth.models.MICRON
    models/pyhealth.models.SafeDrug
    models/pyhealth.models.MoleRec
    models/pyhealth.models.Deepr
    models/pyhealth.models.EHRMamba
    models/pyhealth.models.JambaEHR
    models/pyhealth.models.ContraWR
    models/pyhealth.models.LambdaResNet18ECG
    models/pyhealth.models.ResNet18ECG
    models/pyhealth.models.SEResNet50ECG
    models/pyhealth.models.SparcNet
    models/pyhealth.models.StageNet
    models/pyhealth.models.StageAttentionNet
    models/pyhealth.models.AdaCare
    models/pyhealth.models.ConCare
    models/pyhealth.models.Agent
    models/pyhealth.models.GRASP
    models/pyhealth.models.MedLink
    models/pyhealth.models.TCN
    models/pyhealth.models.TFMTokenizer
    models/pyhealth.models.GAN
    models/pyhealth.models.VAE
    models/pyhealth.models.SDOH
    models/pyhealth.models.VisionEmbeddingModel
    models/pyhealth.models.TextEmbedding
    models/pyhealth.models.BIOT
    models/pyhealth.models.unified_multimodal_embedding_docs
