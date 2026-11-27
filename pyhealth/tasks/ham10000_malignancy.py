"""Malignant vs. benign HAM10000 task with sensitive attributes (MEDFAIR)."""

import itertools
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

MALIGNANT = frozenset({"MEL", "AKIEC"})


def _age_labels(edges: tuple[int, ...]) -> list[str]:
    inner = [f"({lo}, {hi}]" for lo, hi in itertools.pairwise(edges)]
    return [f"<={edges[0]}", *inner, f">{edges[-1]}"]


class HAM10000MalignancyClassification(BaseTask):
    """Binary malignancy classification of HAM10000 images, with subgroups.

    Works with :class:`~pyhealth.datasets.HAM10000Dataset` when the Dataverse
    ``HAM10000_metadata.csv`` is in its root, since age and sex come from it;
    without it no samples are produced. That file covers only the training
    images, so the ISIC 2018 validation and test splits give no samples
    either; split the training set instead, as below.

    The HAM10000 setup of MEDFAIR (Zong et al., ICLR 2023): melanoma (MEL)
    and actinic keratosis / intraepithelial carcinoma (AKIEC) are malignant,
    the other five diagnoses benign, following Maron et al. (2019). Images
    with an unknown age or sex are dropped, leaving 9,948 of the 10,015
    training images. MEDFAIR splits them 80/10/10 by lesion;
    ``PatientSplit((0.8, 0.1, 0.1))`` follows the same protocol. About 14%
    of the images are malignant, so report AUC rather than accuracy.

    Each sample also has:

    - ``sex``: ``"F"`` or ``"M"``.
    - ``age``: age in years, rounded to 5 in HAM10000.
    - ``age_group``: the bin of ``age`` given by ``age_bins``, each bin
      including its upper edge. The default ``(60,)`` gives ``"<=60"`` and
      ``">60"``, MEDFAIR's binary age attribute. Its five-group attribute
      (0-19, 20-39, 40-59, 60-79, 80+) is ``(19, 39, 59, 79)``.

    Args:
        image_size: Side length of the square input image. Defaults to 224.
        age_bins: Increasing upper edges of the age groups. Defaults to
            ``(60,)``.

    Examples:
        >>> from pyhealth.datasets import HAM10000Dataset, PatientSplit
        >>> from pyhealth.tasks import HAM10000MalignancyClassification
        >>> dataset = HAM10000Dataset(root="/path/to/isic2018")  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     HAM10000MalignancyClassification(),
        ...     split=PatientSplit((0.8, 0.1, 0.1), seed=0),
        ... )
        >>> HAM10000MalignancyClassification(age_bins=(19, 39, 59, 79)).age_groups
        ['<=19', '(19, 39]', '(39, 59]', '(59, 79]', '>79']
    """

    task_name: str = "HAM10000MalignancyClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "RGB"})
    }
    output_schema: ClassVar[dict[str, Any]] = {"malignant": "binary"}

    def __init__(self, image_size: int = 224, age_bins: Sequence[int] = (60,)) -> None:
        super().__init__()
        edges = tuple(int(e) for e in age_bins)
        increasing = all(a < b for a, b in itertools.pairwise(edges))
        if not edges or not increasing or edges[0] <= 0:
            raise ValueError(
                f"age_bins must be positive and strictly increasing, got {age_bins!r}."
            )
        self.image_size = int(image_size)
        self.age_bins = edges
        self.age_groups = _age_labels(edges)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per image of the lesion with known age and sex.

        Args:
            patient: A PyHealth ``Patient`` with ``ham10000`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``image`` (the JPEG
            path), ``malignant`` (0 or 1), ``sex``, ``age`` and ``age_group``.
        """
        samples = []
        for event in patient.get_events(event_type="ham10000"):
            sex = getattr(event, "sex", None)
            age = getattr(event, "age", None)
            if sex not in ("F", "M") or age is None or math.isnan(float(age)):
                continue
            age = float(age)
            group = int(np.searchsorted(self.age_bins, age, side="left"))
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "image_id": event.image_id,
                    "image": event.path,
                    "malignant": int(event.diagnosis in MALIGNANT),
                    "sex": sex,
                    "age": age,
                    "age_group": self.age_groups[group],
                }
            )
        return samples
