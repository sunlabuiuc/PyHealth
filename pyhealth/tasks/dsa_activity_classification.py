"""Activity classification task for the DSA dataset.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Turns each 5-second DSA segment into one sample. The target
    sensor (the one worn in daily life) is the model input; the other sensors
    are exposed as synchronized source streams for transfer learning.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

from .base_task import BaseTask


class DSAActivityClassification(BaseTask):
    """19-class activity classification on the DSA dataset.

    Each sample holds the target sensor stream under ``"signal"`` and, for
    transfer learning, each source sensor stream under ``"source_<name>"``.
    All streams of a sample come from the same 5-second window, so
    source/target pairs stay aligned.

    Args:
        target_sensor: Sensor available in daily use. One of ``"torso"``,
            ``"right_arm"``, ``"left_arm"``, ``"right_leg"``, ``"left_leg"``.
        source_sensors: Sensors used as source domains. Defaults to every
            sensor except the target. May be empty for target-only training.

    Attributes:
        task_name (str): Name of the task, including the sensor setup.
        input_schema (dict[str, str]): ``"signal"`` plus one
            ``"source_<name>"`` entry per source sensor, all ``"tensor"``.
        output_schema (dict[str, str]): ``"label"`` as ``"multiclass"``.

    Examples:
        >>> from pyhealth.tasks import DSAActivityClassification
        >>> task = DSAActivityClassification(target_sensor="right_arm")
        >>> task.target_sensor
        'right_arm'
        >>> len(task.input_schema)  # target plus four source sensors
        5
        >>> task.output_schema
        {'label': 'multiclass'}
    """

    task_name: str = "DSAActivityClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

    def __init__(
        self,
        target_sensor: str = "right_arm",
        source_sensors: Sequence[str] | None = None,
    ) -> None:
        # Imported here to avoid a datasets <-> tasks import cycle.
        from pyhealth.datasets.dsa import SENSOR_NAMES

        if target_sensor not in SENSOR_NAMES:
            raise ValueError(
                f"Unknown target sensor '{target_sensor}'. Choose from "
                f"{SENSOR_NAMES}."
            )
        if source_sensors is None:
            source_sensors = [s for s in SENSOR_NAMES if s != target_sensor]
        sources = tuple(source_sensors)
        for name in sources:
            if name not in SENSOR_NAMES:
                raise ValueError(
                    f"Unknown source sensor '{name}'. Choose from {SENSOR_NAMES}."
                )
            if name == target_sensor:
                raise ValueError("The target sensor cannot also be a source.")
        if len(set(sources)) != len(sources):
            raise ValueError("Duplicate source sensors.")

        self.target_sensor = target_sensor
        self.source_sensors = sources
        self.task_name = f"DSAActivityClassification-{target_sensor}-" + (
            "+".join(sources) or "none"
        )
        self.input_schema = {
            "signal": "tensor",
            **{f"source_{name}": "tensor" for name in sources},
        }
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per segment of a subject.

        Args:
            patient: A patient of a :class:`~pyhealth.datasets.DSADataset`
                whose events have ``signal_file``, ``activity`` and
                ``segment`` attributes.

        Returns:
            A list of dicts with ``patient_id``, ``segment``, ``signal``
            (target sensor, shape ``(125, 9)``), one ``source_<name>`` array
            of the same shape per source sensor, and ``label`` (the activity
            number minus one, so labels run from 0 to 18).

        Examples:
            >>> from pyhealth.datasets import DSADataset
            >>> from pyhealth.tasks import DSAActivityClassification
            >>> dataset = DSADataset(root="/data/dsa")  # doctest: +SKIP
            >>> samples = dataset.set_task(
            ...     DSAActivityClassification()
            ... )  # doctest: +SKIP
        """
        from pyhealth.datasets.dsa import (
            SENSOR_NAMES,
            load_dsa_segment,
            split_sensors,
        )

        position = {name: i for i, name in enumerate(SENSOR_NAMES)}
        samples = []
        for event in patient.get_events(event_type="dsa"):
            sensors = split_sensors(load_dsa_segment(event.signal_file))
            sample: dict[str, Any] = {
                "patient_id": patient.patient_id,
                "segment": int(event.segment),
                "signal": sensors[position[self.target_sensor]],
                "label": int(event.activity) - 1,
            }
            for name in self.source_sensors:
                sample[f"source_{name}"] = sensors[position[name]]
            samples.append(sample)
        return samples