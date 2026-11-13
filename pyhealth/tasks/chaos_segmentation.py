"""Official CHAOS CT/MRI segmentation tasks.

The challenge defines five separate tasks.  Each task is a separate PyHealth
task class because each one fixes its own modalities, label set and output
semantics:

- :class:`CHAOSLiverSegmentationCTMRI`: official task 1, liver, CT + MRI.
- :class:`CHAOSLiverSegmentationCTonly`: official task 2, liver, CT only.
- :class:`CHAOSLiverSegmentationMRIonly`: official task 3, liver, MRI only.
- :class:`CHAOSAbdominalSegmentationCTMRI`: official task 4, abdominal organs,
  CT + MRI.  CT annotates the liver only; MRI annotates liver, spleen and both
  kidneys.
- :class:`CHAOSAbdominalSegmentationMRIonly`: official task 5, abdominal
  organs, MRI only.

The input is always a full 3D series.  ``target_spacing`` and ``target_shape``
resample the image and mask together; no 2D conversion happens in the task.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

from .base_task import BaseTask

# Raw CHAOS label conventions (see Train_Sets/NOTES_PLEASE_READ.txt):
#   CT:  1-bit PNGs (0 background, 1 liver via np.asarray; 255 if 8-bit).
#   MRI: 0 background, 63 liver, 126 right kidney, 189 left kidney, 252 spleen.
_LIVER_LABEL_MAP = {
    0: 0,
    1: 1,
    63: 1,
    126: 0,
    189: 0,
    252: 0,
    255: 1,
}
_ORGAN_LABEL_MAP = {
    0: 0,
    1: 1,
    63: 1,
    126: 3,
    189: 4,
    252: 2,
    255: 1,
}


class _CHAOSSegmentationBase(BaseTask):
    """Shared DICOM-series and image-mask sample builder for CHAOS.

    Subclasses set ``task_name``, ``official_task``, ``modalities``,
    ``num_classes``, ``label_map`` and ``class_names``.  ``class_names`` lists
    foreground classes only; class index 0 is always background.

    Args:
        normalize: Intensity normalisation. ``"auto"`` selects CT or MRI from
            each DICOM series' modality.
        target_spacing: Optional target spacing in ``(D, H, W)`` order.
            Applied to both the DICOM series and the mask.
        target_shape: Optional target shape in ``(D, H, W)`` order. Applied to
            both the DICOM series and the mask.
        sequence: Optional additional filter such as ``"T1DUAL"`` or
            ``"T2SPIR"``.
        phase: Optional additional filter such as ``"InPhase"`` or
            ``"OutPhase"``.
        num_classes: Optional override for the number of classes.
        label_map: Optional override for the raw-mask-to-class mapping.

    Examples:
        >>> from pyhealth.tasks import CHAOSLiverSegmentationCTonly
        >>> task = CHAOSLiverSegmentationCTonly()
        >>> task.modalities
        ('CT',)
        >>> task.num_classes
        2
    """

    task_name: ClassVar[str] = "CHAOSSegmentation"
    official_task: ClassVar[int] = 0
    modalities: ClassVar[tuple[str, ...]] = ()
    num_classes: ClassVar[int] = 1
    label_map: ClassVar[dict[int, int]] = {}
    class_names: ClassVar[tuple[str, ...]] = ()
    # Optional per-modality supervision mask.  ``None`` means every class is
    # supervised for every series.
    class_mask_by_modality: ClassVar[
        dict[str, tuple[int, ...]] | None
    ] = None

    def __init__(
        self,
        normalize: str = "auto",
        target_spacing: Sequence[float] | None = None,
        target_shape: Sequence[int] | None = None,
        sequence: str | None = None,
        phase: str | None = None,
        num_classes: int | None = None,
        label_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        if not type(self).modalities:
            raise TypeError(
                "Use one of the public CHAOS task classes, not the base class."
            )
        self.official_task = type(self).official_task
        self.modalities = type(self).modalities
        self.num_classes = (
            int(num_classes)
            if num_classes is not None
            else type(self).num_classes
        )
        self.label_map = dict(
            type(self).label_map if label_map is None else label_map
        )
        self.class_names = type(self).class_names
        self.class_mask_by_modality = type(self).class_mask_by_modality
        if self.label_map and max(self.label_map.values()) >= self.num_classes:
            raise ValueError(
                "label_map contains a class index outside num_classes; "
                "pass matching num_classes and label_map together."
            )
        if self.class_mask_by_modality is not None:
            for modality, mask in self.class_mask_by_modality.items():
                if len(mask) != self.num_classes:
                    raise ValueError(
                        f"class_mask for {modality} has {len(mask)} entries "
                        f"but num_classes={self.num_classes}."
                    )
        self.sequence = sequence
        self.phase = phase

        image_kwargs: dict[str, Any] = {"lazy": True, "normalize": normalize}
        mask_kwargs: dict[str, Any] = {
            "lazy": True,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            # CHAOS masks are mask_N.png while DICOM slices have their own
            # names, so pair them positionally in natural filename order.
            "strict_mask_names": False,
        }
        if target_spacing is not None:
            spacing = tuple(target_spacing)
            image_kwargs["target_spacing"] = spacing
            mask_kwargs["target_spacing"] = spacing
        if target_shape is not None:
            shape = tuple(target_shape)
            image_kwargs["target_shape"] = shape
            mask_kwargs["target_shape"] = shape

        self.input_schema = {"image": ("dicom_series", image_kwargs)}
        self.output_schema = {"mask": ("image_mask_dir", mask_kwargs)}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per labelled series allowed by this task.

        Args:
            patient: A PyHealth ``Patient`` with ``series`` events.

        Returns:
            A list of sample dictionaries. ``image`` is the DICOM series path
            and ``mask`` is a ``(mask_dir, series_path)`` tuple used to align
            image slices with the DICOM slice order.
        """
        samples = []
        for event in patient.get_events(event_type="series"):
            if not getattr(event, "mask_dir", None):
                continue
            event_modality = str(getattr(event, "modality", ""))
            if event_modality not in self.modalities:
                continue
            if self.sequence and str(getattr(event, "sequence", "")) != self.sequence:
                continue
            if self.phase and str(getattr(event, "phase", "")) != self.phase:
                continue
            sample = {
                "patient_id": patient.patient_id,
                "record_id": event.record_id,
                "image": event.series_path,
                "mask": (event.mask_dir, event.series_path),
                "modality": getattr(event, "modality", None),
                "sequence": getattr(event, "sequence", None),
                "phase": getattr(event, "phase", None),
                "chaos_task": self.task_name,
            }
            if self.class_mask_by_modality is not None:
                class_mask = self.class_mask_by_modality.get(event_modality)
                if class_mask is None:
                    raise ValueError(
                        f"No class_mask for modality {event_modality!r} in "
                        f"{self.task_name}."
                    )
                sample["class_mask"] = list(class_mask)
            samples.append(sample)
        return samples


class CHAOSLiverSegmentationCTMRI(_CHAOSSegmentationBase):
    """Official CHAOS task 1: liver segmentation from CT and MRI.

    This is the official cross-modality liver task.  A single model sees CT and
    MR series as separate single-channel samples; they are not registered or
    concatenated as paired channels.

    Examples:
        >>> from pyhealth.tasks import CHAOSLiverSegmentationCTMRI
        >>> CHAOSLiverSegmentationCTMRI().modalities
        ('CT', 'MR')
    """

    task_name = "CHAOSLiverSegmentationCTMRI"
    official_task = 1
    modalities = ("CT", "MR")
    num_classes = 2
    label_map = _LIVER_LABEL_MAP
    class_names = ("liver",)


class CHAOSLiverSegmentationCTonly(_CHAOSSegmentationBase):
    """Official CHAOS task 2: liver segmentation from CT only.

    Examples:
        >>> from pyhealth.tasks import CHAOSLiverSegmentationCTonly
        >>> CHAOSLiverSegmentationCTonly().modalities
        ('CT',)
    """

    task_name = "CHAOSLiverSegmentationCTonly"
    official_task = 2
    modalities = ("CT",)
    num_classes = 2
    label_map = _LIVER_LABEL_MAP
    class_names = ("liver",)


class CHAOSLiverSegmentationMRIonly(_CHAOSSegmentationBase):
    """Official CHAOS task 3: liver segmentation from MRI only.

    MRI includes T1-DUAL in/out phase and T2-SPIR series.

    Examples:
        >>> from pyhealth.tasks import CHAOSLiverSegmentationMRIonly
        >>> CHAOSLiverSegmentationMRIonly().modalities
        ('MR',)
    """

    task_name = "CHAOSLiverSegmentationMRIonly"
    official_task = 3
    modalities = ("MR",)
    num_classes = 2
    label_map = _LIVER_LABEL_MAP
    class_names = ("liver",)


class CHAOSAbdominalSegmentationCTMRI(_CHAOSSegmentationBase):
    """Official CHAOS task 4: abdominal organs from CT and MRI.

    CT annotates the liver only.  MRI annotates liver, spleen and both kidneys,
    so the shared output has five classes including background.

    Examples:
        >>> from pyhealth.tasks import CHAOSAbdominalSegmentationCTMRI
        >>> CHAOSAbdominalSegmentationCTMRI().num_classes
        5
    """

    task_name = "CHAOSAbdominalSegmentationCTMRI"
    official_task = 4
    modalities = ("CT", "MR")
    num_classes = 5
    label_map = _ORGAN_LABEL_MAP
    class_names = ("liver", "spleen", "right kidney", "left kidney")
    # CT annotates the liver only.  Supervise background + liver and leave the
    # unannotated spleen/kidney classes out of the loss instead of teaching the
    # model that those organs are background on CT.
    class_mask_by_modality: ClassVar[dict[str, tuple[int, ...]]] = {
        "CT": (1, 1, 0, 0, 0),
        "MR": (1, 1, 1, 1, 1),
    }


class CHAOSAbdominalSegmentationMRIonly(_CHAOSSegmentationBase):
    """Official CHAOS task 5: abdominal organs from MRI only.

    Examples:
        >>> from pyhealth.tasks import CHAOSAbdominalSegmentationMRIonly
        >>> CHAOSAbdominalSegmentationMRIonly().class_names
        ('liver', 'spleen', 'right kidney', 'left kidney')
    """

    task_name = "CHAOSAbdominalSegmentationMRIonly"
    official_task = 5
    modalities = ("MR",)
    num_classes = 5
    label_map = _ORGAN_LABEL_MAP
    class_names = ("liver", "spleen", "right kidney", "left kidney")
