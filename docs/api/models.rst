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
   * - ``RNN_V2``
     - Features are ``unified`` fields (nested codes, values and timestamps); static values are a series with one record at ``t = 0``
     - Encodes each field bottom-up over packed records, without padding the hierarchy
   * - :doc:`models/pyhealth.models.StageNet`
     - You have time-stamped vital signs with irregular measurement intervals
     - Requires ``StageNetProcessor`` or ``StageNetTensorProcessor`` in the task schema
   * - :doc:`models/pyhealth.models.UNet`
     - Your task is pixel/voxel-level segmentation with an image and a mask
     - ``UNet2D``/``UNet3D``; pair with ``PatchDataset``, ``PatchSampler`` and Dice/IoU metrics
   * - :doc:`models/pyhealth.models.GNN`
     - Features include graph-structured data
     - Works with ``GraphProcessor``; see :doc:`graph` for setup
   * - :doc:`models/pyhealth.models.GraphCare`
     - You want to augment EHR codes with a medical knowledge graph
     - Combines code sequences with a :class:`~pyhealth.graph.KnowledgeGraph`
   * - :doc:`models/pyhealth.models.XGBoostModel`
     - Features are tabular (engineered labs, demographics) or bags of codes; you want the standard strong baseline
     - Gradient-boosted trees; fit with ``model.fit`` (not ``Trainer.train``); exact TreeSHAP; needs ``pip install "pyhealth[xgboost]"``

How BaseModel Works
--------------------

All PyHealth models inherit from ``BaseModel``, which itself inherits from
PyTorch's ``nn.Module``. When you call ``MyModel(dataset=sample_ds)``, the
base class reads the dataset's schemas and automatically sets:

- ``self.feature_keys`` — the list of input field names from ``input_schema``
- ``self.label_keys`` — the list of output field names from ``output_schema``
- ``self.device`` — the compute device

It also gives the label processor, ``self.label_processor`` (see
:class:`~pyhealth.processors.LabelProcessor`), which takes care of everything
that varies by task type:

- ``self.label_processor.output_size()`` is the output dimension, so you don't have
  to hard-code it.
- ``self.label_processor.loss(logits, y_true)`` is the right loss for the
  task: BCE for binary and multilabel tasks, cross-entropy for multiclass, MSE
  for regression.
- ``self.label_processor.activation(logits)`` applies sigmoid, softmax, or
  identity to logits depending on the task, producing ``y_prob``.

A model with several labels uses ``self.label_processors``, a dict by output
key.

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
   * - ``"timeseries_binary"``, ``"timeseries_multiclass"``,
       ``"timeseries_multilabel"``, ``"timeseries_regression"``
     - the static loss, per label record
     - the static activation → (label records, ...)

The schema value can be written as a string, a label processor class, a
processor instance, or a ``(name, kwargs)`` tuple. These are equivalent:

.. code-block:: python

    output_schema = {"icd_codes": "multilabel"}
    output_schema = {"icd_codes": MultiLabelProcessor}
    output_schema = {"icd_codes": ("multilabel", {})}

Whichever form a task uses, the model gets the same fitted label processor,
and ``model.mode`` is its kind (``"binary"``, ``"multiclass"``, ...), or
``None`` when the model has no single label processor. ``mode`` is read-only.
``Trainer.evaluate()`` asks the label processor how to gather the predictions
and which metrics to compute. A model that scores something other than its
dataset's label (for example a reconstruction) overrides
``label_processors``.

Segmentation
^^^^^^^^^^^^

For pixel/voxel-level tasks, use :doc:`models/pyhealth.models.UNet` together
with :class:`~pyhealth.datasets.PatchDataset` and
:class:`~pyhealth.sampler.PatchSampler`. ``num_classes=1`` selects binary
segmentation; larger values select multiclass segmentation.

.. code-block:: python

    from pyhealth.datasets import PatchDataset, PatientSplit, get_dataloader
    from pyhealth.models import UNet3D
    from pyhealth.sampler import PatchSampler

    train, val = dataset.set_task(
        dataset.default_task,
        split=PatientSplit((0.8, 0.2), seed=42),
    )
    patches = PatchDataset(
        train,
        PatchSampler((64, 64, 64), foreground_prob=1 / 3),
        mode="train",
    )
    model = UNet3D(train)  # original U-Net defaults: 64 channels, depth 5
    loader = get_dataloader(patches, batch_size=2, shuffle=True)

``UNet`` reports ``"segmentation_binary"`` or ``"segmentation_multiclass"``
from ``num_classes`` (or the mask processor's configured ``num_classes``), so the
``Trainer`` can score variable-shaped masks with Dice and IoU.

A task can add a boolean ``class_mask`` vector to each sample to mark which
classes are annotated. ``SegmentationLoss`` excludes unannotated classes from
the loss, and ``evaluate_segmentation`` remaps the metrics to the kept classes.
``PatchDataset`` does not create ``class_mask``; the task supplies it.

A class is included in the Dice/IoU average when either the target or the
prediction has foreground. If the target is empty but the prediction has
foreground, that class scores 0.0; classes empty in both are left out. The
case score is NaN only when no class has foreground in either the target or
the prediction.

``Trainer.evaluate`` uses the model's label processor and does not read a
per-sample ``class_mask``. For partially annotated data, evaluate with
:func:`~pyhealth.models.evaluate_segmentation`, which remaps each case to its
annotated classes before scoring.

When ``in_channels``, ``num_classes`` or ``spatial_dim`` are omitted, ``UNet``
reads them from the image and mask processors or the first materialised sample
(a ``{0, 1}`` mask without ``num_classes`` infers ``num_classes=2`` multiclass;
pass ``num_classes=1`` for single-channel binary segmentation). Lazy values are
materialised before their shape or labels are inspected.

Imbalanced outcomes
^^^^^^^^^^^^^^^^^^^

For rare binary or multilabel outcomes, the task weights the positive examples
in the loss with the label processor's ``pos_weight``:

.. code-block:: python

    output_schema = {"mortality": ("binary", {"pos_weight": "balanced"})}
    # or a fixed weight: ("binary", {"pos_weight": 9.0}); per label, in the
    # order of the given labels:
    # ("multilabel", {"labels": [...], "pos_weight": [...]})

``"balanced"`` is negatives / positives over the samples the processor is
fitted on, so fit the processors on the training split
(``set_task(task, split=PatientSplit(...))``); PyHealth warns otherwise. The
weight applies to models that use ``label_processor.loss``; the
drug-recommendation models (GAMENet, SafeDrug, MoleRec, MICRON, ARMR) compute
their own loss and ignore it. Weighting shifts the predicted probabilities
upward, so check calibration before interpreting them as risks. See
``examples/imbalanced_outcome_pos_weight.py``.

Building a Custom Model
-----------------------

If none of the built-in models fit your architecture, you can subclass
``BaseModel`` directly. The skeleton below shows the typical structure: build
an ``EmbeddingModel`` in ``__init__``, unpack processor schemas in
``forward``, pool or aggregate the embeddings, and return ``logit`` and
``y_prob``, plus ``loss`` and ``y_true`` when the batch has a label.

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
                                self.label_processor.output_size())

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

            results = {
                "y_prob": self.label_processor.activation(logits),
                "logit": logits,
            }
            # The label is absent at inference time; add the loss only when
            # it is present, as the built-in models do.
            if self.label_key in kwargs:
                y_true = kwargs[self.label_key].to(self.device)
                results["loss"] = self.label_processor.loss(logits, y_true)
                results["y_true"] = y_true
            return results

API Reference
-------------

.. toctree::
    :maxdepth: 3

    models/pyhealth.models.BaseModel
    models/pyhealth.models.LogisticRegression
    models/pyhealth.models.MLP
    models/pyhealth.models.CNN
    models/pyhealth.models.UNet
    models/pyhealth.models.RNN
    models/pyhealth.models.GNN
    models/pyhealth.models.Transformer
    models/pyhealth.models.TransformersModel
    models/pyhealth.models.LLMEHREncoder
    models/pyhealth.models.MEME
    models/pyhealth.models.TransformerDeID
    models/pyhealth.models.RETAIN
    models/pyhealth.models.GAMENet
    models/pyhealth.models.GraphCare
    models/pyhealth.models.MICRON
    models/pyhealth.models.SafeDrug
    models/pyhealth.models.SHy
    models/pyhealth.models.MoleRec
    models/pyhealth.models.ARMR
    models/pyhealth.models.Deepr
    models/pyhealth.models.EHRMamba
    models/pyhealth.models.EHRMambaCEHR
    models/pyhealth.models.JambaEHR
    models/pyhealth.models.ContraWR
    models/pyhealth.models.SparcNet
    models/pyhealth.models.StageNet
    models/pyhealth.models.StageAttentionNet
    models/pyhealth.models.AdaCare
    models/pyhealth.models.ConCare
    models/pyhealth.models.Agent
    models/pyhealth.models.GRASP
    models/pyhealth.models.MedFuse
    models/pyhealth.models.MedLink
    models/pyhealth.models.TCN
    models/pyhealth.models.TFMTokenizer
    models/pyhealth.models.GAN
    models/pyhealth.models.VAE
    models/pyhealth.models.HALO
    models/pyhealth.models.GPT2
    models/pyhealth.models.PromptEHR
    models/pyhealth.models.MedGAN
    models/pyhealth.models.CorGAN
    models/pyhealth.models.Synthea
    models/pyhealth.models.SDOH
    models/pyhealth.models.SequenceEmbeddingModel
    models/pyhealth.models.VisionEmbeddingModel
    models/pyhealth.models.TextEmbedding
    models/pyhealth.models.BIOT
    models/pyhealth.models.unified_multimodal_embedding_docs
    models/pyhealth.models.califorest
    models/pyhealth.models.XGBoostModel
    models/pyhealth.models.survival_mdn
    models/pyhealth.models.TrajSurv
    models/pyhealth.models.KnowRare
    models/pyhealth.models.APRICOTM
    models/pyhealth.models.ProtoECGNet
    models/pyhealth.models.HeartLang
    models/pyhealth.models.APN
    models/pyhealth.models.TRACE
    models/pyhealth.models.MIRA

