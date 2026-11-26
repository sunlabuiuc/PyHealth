from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor

from .base_model import BaseModel

FEATURES = ("epoch_feature", "sequence_feature")


class SleepKD(BaseModel):
    """SleepKD: multi-level knowledge distillation for sleep staging.

    Paper: Heng Liang, Yucheng Liu, Haichao Wang, and Ziyu Jia. "Teacher
    Assistant-Based Knowledge Distillation Extracting Multi-level Features on
    Single Channel Sleep EEG." IJCAI 2023, pp. 3948-3956.

    Trains ``student`` from a frozen ``teacher`` with the loss of eq. 5::

        L = alpha * L_epoch + beta * L_seq + gamma * L_soft + delta * L_hard

    - ``L_epoch`` and ``L_seq`` (eq. 1-2) compare the teacher's epoch and
      sequence features, aligned to the student's sizes, with the student's.
    - ``L_soft`` (eq. 3) is ``KL(p_T || p_S)`` between the teacher's and the
      student's per-epoch class distributions.
    - ``L_hard`` (eq. 4) is the cross-entropy with the true stages.

    Both models need an ``encode(x)`` method returning ``epoch_feature``
    ``(batch, seq_len, d_e)``, ``sequence_feature`` ``(batch, seq_len, d_s)``
    and ``logit`` ``(batch, seq_len, n_classes)``, as
    :class:`~pyhealth.models.DeepSleepNet` does. The teacher-assistant scheme
    of the paper is two runs: teacher to assistant, then the trained assistant
    (as the teacher) to the student. Setting weights to zero gives baselines:
    ``alpha=beta=0`` is KD (Hinton et al., 2015) with hard labels, and the same
    weights through a teacher assistant are TAKD (Mirzadeh et al., 2020).

    The teacher is frozen and always in eval mode. The model's outputs are the
    student's, so :class:`~pyhealth.trainer.Trainer` scores the student.

    Args:
        dataset: The dataset to train the model, the one of both models.
        teacher: The trained teacher (or teacher assistant).
        student: The model to train.
        alpha: Weight of ``L_epoch``. Default is 1.0.
        beta: Weight of ``L_seq``. Default is 0.1.
        gamma: Weight of ``L_soft``. Default is 1.0.
        delta: Weight of ``L_hard``. Default is 1.0. The defaults are the
            paper's weights for DeepSleepNet; for SalientSleepNet it uses
            0.3, 0.2, 0.4 and 0.1.
        feature_loss: ``"mse"`` (the paper, eq. 1-2) or ``"kl"``, the KL
            divergence between softmaxes over the feature entries of each
            epoch, averaged over epochs like ``L_soft`` (the official code
            uses a KL divergence). Default is ``"mse"``.
        align: The alignment of the teacher's features to the student's
            sizes. ``"maxpool"`` max-pools each feature vector to the
            student's size with ``F.adaptive_max_pool1d`` (no parameters),
            whose windows are disjoint when the student's size divides the
            teacher's and may overlap otherwise;
            ``"slidingmax"`` takes the maximum of every ``d_t - d_s + 1``
            consecutive entries, the alignment of the authors' code
            (https://github.com/AI4Brain/SleepKD), also without parameters;
            ``"conv"`` is a learned 1x1 convolution over the feature entries
            of each epoch, trained with the student. Default is
            ``"maxpool"``.
        temperature: Softmax temperature of ``L_soft``, which is scaled by
            ``temperature ** 2`` (Hinton et al., 2015). The paper gives none,
            i.e. 1. Default is 1.0.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import DeepSleepNet, SleepKD
        >>> samples = [
        ...     {
        ...         "patient_id": str(i),
        ...         "signal": np.random.randn(4, 1, 3000).astype("float32"),
        ...         "label": np.array([0, 1, 2, 4]),
        ...     }
        ...     for i in range(2)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": ("tensor", {"dtype": torch.long})},
        ... )
        >>> teacher = DeepSleepNet(dataset=dataset, n_conv=2, hidden_dim=32)
        >>> student = DeepSleepNet(dataset=dataset, n_conv=1, hidden_dim=16)
        >>> model = SleepKD(dataset=dataset, teacher=teacher, student=student)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> output = model(**batch)
        >>> sorted(k for k in output if k.startswith("loss_"))
        ['loss_epoch', 'loss_hard', 'loss_seq', 'loss_soft']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        teacher: BaseModel,
        student: BaseModel,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 1.0,
        delta: float = 1.0,
        feature_loss: Literal["mse", "kl"] = "mse",
        align: Literal["maxpool", "slidingmax", "conv"] = "maxpool",
        temperature: float = 1.0,
    ):
        super().__init__(dataset=dataset)
        for name, model in (("teacher", teacher), ("student", student)):
            if not callable(getattr(model, "encode", None)):
                raise TypeError(
                    f"The {name} needs an encode(x) method returning "
                    f"{', '.join(FEATURES)} and logit, as DeepSleepNet does."
                )
        weights = {"epoch": alpha, "seq": beta, "soft": gamma, "hard": delta}
        if min(weights.values()) < 0 or not any(weights.values()):
            raise ValueError(
                f"The loss weights must be non-negative, not all zero; got {weights}."
            )
        if feature_loss not in ("mse", "kl"):
            raise ValueError(
                f"feature_loss must be 'mse' or 'kl', got {feature_loss!r}."
            )
        if align not in ("maxpool", "slidingmax", "conv"):
            raise ValueError(
                f"align must be 'maxpool', 'slidingmax' or 'conv', got {align!r}."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.teacher = teacher
        self.student = student
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.weights = weights
        self.feature_loss = feature_loss
        self.align = align
        self.temperature = temperature
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]

        # Feature sizes, from one sample, for the alignment.
        x = dataset[0][self.feature_key][None]
        with torch.no_grad():
            was_training = student.training
            student.eval()
            t_out, s_out = teacher.encode(x), student.encode(x)
            student.train(was_training)
        self.feature_dims = {
            key: (t_out[key].shape[-1], s_out[key].shape[-1]) for key in FEATURES
        }
        if align != "conv":
            for key, (d_t, d_s) in self.feature_dims.items():
                if d_t < d_s:
                    raise ValueError(
                        f"align={align!r} needs teacher features at least as "
                        f"large as the student's; {key} has {d_t} < {d_s}. "
                        "Use align='conv'."
                    )
        self.aligners = nn.ModuleDict(
            {
                key: nn.Conv1d(d_t, d_s, kernel_size=1)
                for key, (d_t, d_s) in self.feature_dims.items()
                if align == "conv"
            }
        )

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The student's label processors: the Trainer scores the student."""
        return self.student.label_processors

    def train(self, mode: bool = True) -> "SleepKD":
        """Sets training mode, keeping the frozen teacher in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def _align(self, key: str, feature: torch.Tensor) -> torch.Tensor:
        """Maps teacher features ``(batch, seq_len, d_t)`` to the student's size."""
        d_t, d_s = self.feature_dims[key]
        if self.align == "conv":
            return self.aligners[key](feature.transpose(1, 2)).transpose(1, 2)
        if d_t == d_s:
            return feature
        batch, seq_len, _ = feature.shape
        flat = feature.reshape(-1, 1, d_t)
        if self.align == "slidingmax":
            pooled = F.max_pool1d(flat, kernel_size=d_t - d_s + 1, stride=1)
        else:
            pooled = F.adaptive_max_pool1d(flat, d_s)
        return pooled.view(batch, seq_len, d_s)

    def _feature_distance(
        self, teacher: torch.Tensor, student: torch.Tensor
    ) -> torch.Tensor:
        if self.feature_loss == "mse":
            return F.mse_loss(student, teacher)
        # One distribution per epoch, so the mean is over epochs, as in L_soft.
        return F.kl_div(
            F.log_softmax(student.flatten(0, -2), dim=-1),
            F.log_softmax(teacher.flatten(0, -2), dim=-1),
            reduction="batchmean",
            log_target=True,
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Returns:
            A dict with the student's ``y_prob`` and ``logit`` ``(batch *
            seq_len, n_classes)``, ``y_true`` ``(batch * seq_len,)``, the total
            ``loss``, and the unweighted terms ``loss_epoch``, ``loss_seq``,
            ``loss_soft`` and ``loss_hard``.
        """
        x = kwargs[self.feature_key].to(self.device)
        y_true = kwargs[self.label_key].to(self.device).flatten()
        with torch.no_grad():
            t_out = self.teacher.encode(x)
        s_out = self.student.encode(x)

        s_logit = s_out["logit"].flatten(0, 1)
        t_logit = t_out["logit"].flatten(0, 1)
        T = self.temperature
        terms = {
            "epoch": self._feature_distance(
                self._align("epoch_feature", t_out["epoch_feature"]),
                s_out["epoch_feature"],
            ),
            "seq": self._feature_distance(
                self._align("sequence_feature", t_out["sequence_feature"]),
                s_out["sequence_feature"],
            ),
            "soft": F.kl_div(
                F.log_softmax(s_logit / T, dim=-1),
                F.log_softmax(t_logit / T, dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            * T**2,
            "hard": F.cross_entropy(s_logit, y_true),
        }
        loss = sum(self.weights[k] * v for k, v in terms.items() if self.weights[k])
        label_processor = next(iter(self.label_processors.values()))
        results = {
            "loss": loss,
            "y_prob": label_processor.activation(s_logit),
            "y_true": y_true,
            "logit": s_logit,
        }
        results.update({f"loss_{k}": v.detach() for k, v in terms.items()})
        return results
