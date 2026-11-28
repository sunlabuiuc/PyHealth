pyhealth.models.GLIP
====================

GLIP-T, a grounded language-image detector: it finds the phrases of a text
prompt in an image, scoring every anchor box against every prompt token
(`Li et al., CVPR 2022 <https://arxiv.org/abs/2112.03857>`_,
`official code <https://github.com/microsoft/GLIP>`_, MIT licence). MIU-VL
(`Qin et al., ICLR 2023 <https://arxiv.org/abs/2209.15517>`_,
`code <https://github.com/MembrAI/MIU-VL>`_) uses it zero-shot on medical
images, with prompts that describe the target's colour, shape and location;
:mod:`pyhealth.models.miu_vl` builds those prompts.

This is a pure-PyTorch port of GLIP-T (Swin-T, Objects365 + GoldG). The
official code needs ``maskrcnn_benchmark`` and compiled CUDA kernels; the port
uses ``torchvision.ops`` and runs on CPU, CUDA and Apple MPS.
``GLIP(pretrained=True)`` downloads the official checkpoint
(`GLIPModel/GLIP <https://huggingface.co/GLIPModel/GLIP>`_, pinned revision,
3.7 GB) and loads every weight. On a CVC-300 frame the port gives the same
detections as the official code run on CPU (scores within 3e-7), and the
class-name prompt ``"polyp"`` reproduces the paper's 6.1 AP on CVC-300 and
4.1 AP on CVC-ClinicDB.

The model is zero-shot only: it has no loss and is not fine-tuned. Boxes are
returned in COCO ``[x, y, width, height]`` format, for
:func:`~pyhealth.metrics.detection_metrics_fn`. See
``examples/miu_vl_polyp_zero_shot.py`` for zero-shot polyp detection on
:class:`~pyhealth.datasets.PolypDataset`.

.. autoclass:: pyhealth.models.GLIP
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.GLIPDetector
    :members: forward, load_checkpoint, preprocess
    :show-inheritance:

.. autofunction:: pyhealth.models.build_caption

MIU-VL prompts
--------------

.. automodule:: pyhealth.models.miu_vl
    :members:
