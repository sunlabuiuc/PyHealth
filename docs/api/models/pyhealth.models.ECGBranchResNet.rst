pyhealth.models.ECGBranchResNet
===================================

Black-box, label-grouped ECG ResNet with a learned fusion head. It uses the
same label groups and per-branch backbones as
:doc:`pyhealth.models.ProtoECGNet`, but each branch ends in a plain pooled
linear head, so it isolates the effect of prototypes from the effect of the
branch split and of the fusion layer
(`Sethi et al., MLHC 2025 <https://arxiv.org/abs/2504.08713>`_). With a
single group it is the paper's full-label black-box ResNet baseline.

Typical use with :doc:`../tasks/pyhealth.tasks.PTBXLLabelGroupClassification`:

.. code-block:: python

    from pyhealth.models import ECGBranchResNet
    from pyhealth.tasks.ptbxl_label_group_classification import (
        PHYSICIAN_LABEL_GROUPS,
    )

    model = ECGBranchResNet(
        samples,
        label_groups=PHYSICIAN_LABEL_GROUPS,
        branch_kwargs={
            "morphology": {"backbone": "resnet2d"},
            "global": {"backbone": "resnet2d"},
        },
        fusion="features",
    )
    model.set_stage("branches")  # train the branches, then
    model.set_stage("fusion")    # train only the fusion head

See ``examples/ptbxl_label_group_classification_ecg_branch_resnet.py`` for the
full pipeline and ablation study.

.. autoclass:: pyhealth.models.ECGBranchResNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ResNetBranch
    :members:
    :undoc-members:
    :show-inheritance:
