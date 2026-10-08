from collections.abc import Sequence
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type
from typing import TYPE_CHECKING
import inspect
import random
from bisect import bisect_right
import litdata
from litdata.utilities.train_test_split import deepcopy_dataset
import copy
import logging

import numpy as np
import torch

from ..processors import get_processor, IgnoreProcessor
from ..processors.base_processor import FeatureProcessor

if TYPE_CHECKING:  # splitter imports this module
    from .splitter import Split

logger = logging.getLogger(__name__)

# How processed samples are laid out in the disk (litdata) cache. Bump when
# the layout changes; it is part of the processed-cache key, so caches written
# with an older layout are rebuilt instead of misread.
STORAGE_FORMAT = 2
# Record key holding one pickled payload with every non-tensor field.
_OBJECTS_KEY = "_pyhealth_objects"
_PICKLE_PROTOCOL = 5


def _decode_record(item: Any) -> Any:
    """Restores a sample written by :meth:`SampleBuilder.encode`.

    Records without the objects payload (caches written before
    ``STORAGE_FORMAT`` 2, or in-memory samples) are returned unchanged.
    """
    if isinstance(item, list):
        return [_decode_record(x) for x in item]
    if not isinstance(item, dict) or _OBJECTS_KEY not in item:
        return item
    payload = pickle.loads(item[_OBJECTS_KEY])
    values = payload["values"]
    return {k: values[k] if k in values else item[k] for k in payload["order"]}


class _MaskedSamples:
    """A re-iterable view of a sample stream that keeps only masked positions.

    Each iteration walks the underlying stream in order and yields the samples
    whose position is True in ``mask``. Nothing is collected, so fitting on a
    split costs no more memory than fitting on all samples.
    """

    def __init__(self, samples: Iterable[dict[str, Any]], mask: np.ndarray):
        self._samples = samples
        self._mask = mask

    def __iter__(self):
        for keep, sample in zip(self._mask, self._samples):
            if keep:
                yield sample


def _remap_index_mapping(
    mapping: dict[str, list[int]], indices: Sequence[int]
) -> dict[str, list[int]]:
    key_by_index = {
        index: key
        for key, mapped_indices in mapping.items()
        for index in mapped_indices
    }
    remapped: dict[str, list[int]] = {}
    for new_index, old_index in enumerate(indices):
        key = key_by_index.get(old_index)
        if key is not None:
            remapped.setdefault(key, []).append(new_index)
    return remapped


class SampleBuilder:
    """Fit feature processors and transform pickled samples without materializing a dataset.

    SampleBuilder is a lightweight helper used to:
        - Fit feature processors from provided `input_schema` and `output_schema` on an
            iterable of raw Python sample dictionaries.
        - Build mappings from patient IDs and record IDs to sample indices.
        - Transform pickled sample records into processed feature dictionaries using
            the fitted processors.

    Typical usage:
        builder = SampleBuilder(input_schema, output_schema)
        builder.fit(samples)
        builder.save(path)  # writes a schema.pkl metadata file

    After saving the schema, `litdata.optimize` can be used with `builder.transform`
    to serialize and chunk pickled sample items into a directory that can be
    loaded via SampleDataset.

    Pre-fitted processors passed as `input_processors` / `output_processors` are
    used as they are; every schema field without one gets a fitted processor.

    Examples:
        >>> from pyhealth.datasets.sample_dataset import SampleBuilder
        >>> from pyhealth.processors import SequenceProcessor
        >>> samples = [
        ...     {"patient_id": "p1", "codes": ["a", "b"], "age": [40.0], "label": 1},
        ...     {"patient_id": "p2", "codes": ["c"], "age": [55.0], "label": 0},
        ... ]
        >>> codes = SequenceProcessor()
        >>> codes.fit(samples, "codes")
        >>> builder = SampleBuilder(
        ...     input_schema={"codes": "sequence", "age": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     input_processors={"codes": codes},
        ... )
        >>> builder.fit(samples)
        >>> builder.input_processors["codes"] is codes
        True
        >>> type(builder.input_processors["age"]).__name__
        'TensorProcessor'
    """

    def __init__(
        self,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> None:
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._input_processors = (
            dict(input_processors) if input_processors is not None else {}
        )
        self._output_processors = (
            dict(output_processors) if output_processors is not None else {}
        )
        self._patient_to_index: Dict[str, List[int]] = {}
        self._record_to_index: Dict[str, List[int]] = {}
        self._fitted = False
        self._split = None
        self._split_indices: list[np.ndarray] | None = None

    @property
    def input_processors(self) -> Dict[str, FeatureProcessor]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing input_processors."
            )
        return self._input_processors

    @property
    def output_processors(self) -> Dict[str, FeatureProcessor]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing output_processors."
            )
        return self._output_processors

    @property
    def split_indices(self) -> list[np.ndarray] | None:
        """Sample indices of each split part (train first), or None without a split."""
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing split_indices."
            )
        return self._split_indices

    @property
    def patient_to_index(self) -> Dict[str, List[int]]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing patient_to_index."
            )
        return self._patient_to_index

    @property
    def record_to_index(self) -> Dict[str, List[int]]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing record_to_index."
            )
        return self._record_to_index

    def _get_processor_instance(self, processor_spec):
        """Instantiate a processor using the same resolution logic as SampleDataset."""
        if isinstance(processor_spec, tuple):
            spec, kwargs = processor_spec
            if isinstance(spec, str):
                return get_processor(spec)(**kwargs)
            if inspect.isclass(spec) and issubclass(spec, FeatureProcessor):
                return spec(**kwargs)
            raise ValueError(
                "Processor spec in tuple must be either a string alias or a "
                f"FeatureProcessor class, got {type(spec)}"
            )
        if isinstance(processor_spec, str):
            return get_processor(processor_spec)()
        if inspect.isclass(processor_spec) and issubclass(
            processor_spec, FeatureProcessor
        ):
            return processor_spec()
        if isinstance(processor_spec, FeatureProcessor):
            return processor_spec
        raise ValueError(
            "Processor spec must be either a string alias, a FeatureProcessor "
            f"class, or a tuple (spec, kwargs_dict), got {type(processor_spec)}"
        )

    def fit(
        self,
        samples: Iterable[Dict[str, Any]],
        split: "Split | None" = None,
    ) -> None:
        """Fit processors and build mapping indices from an iterator of samples.

        Args:
            samples: Iterable of sample dictionaries (e.g., python dicts). Each
                sample should contain keys covering both the configured
                `input_schema` and `output_schema`. These samples are not
                required to be pickled; `fit` operates on in-memory dicts.
                It is iterated several times and never collected, so a
                streaming dataset works without loading it into memory.
            split: Optional :class:`~pyhealth.datasets.Split`, such as
                :class:`~pyhealth.datasets.PatientSplit`. When given,
                processors are fitted on the first (training) part only, and
                the sample indices of every part are kept in
                ``split_indices``.

        Behavior:
            - Validates the samples contain all keys specified by the input
              and output schemas.
            - Builds `patient_to_index` and `record_to_index` mappings by
              recording the sample indices associated with `patient_id` and
              `record_id`/`visit_id` fields.
            - Instantiates and fits a processor for every schema field that
              has none. Pre-fitted processors supplied to the constructor are
              used as they are; any remaining fields still get fitted ones.
        """
        # Validate the samples
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for sample in samples:
            assert input_keys.issubset(
                sample.keys()
            ), "Input schema does not match samples."
            assert output_keys.issubset(
                sample.keys()
            ), "Output schema does not match samples."

        # Build index mappings
        self._patient_to_index = {}
        self._record_to_index = {}
        n_samples = 0
        for i, sample in enumerate(samples):
            n_samples = i + 1
            patient_id = sample.get("patient_id")
            if patient_id is not None:
                self._patient_to_index.setdefault(patient_id, []).append(i)
            record_id = sample.get("record_id", sample.get("visit_id"))
            if record_id is not None:
                self._record_to_index.setdefault(record_id, []).append(i)

        # With a split, fit on the training patients only: a masked view of
        # the same stream, so nothing is loaded into memory.
        self._split = split
        self._split_indices = None
        fit_samples = samples
        if split is not None:
            self._split_indices = split.split_indices(self._patient_to_index)
            train_mask = np.zeros(n_samples, dtype=bool)
            train_mask[self._split_indices[0]] = True
            fit_samples = _MaskedSamples(samples, train_mask)

        # Fit a processor for every schema field that has none. Supplied
        # (pre-fitted) processors are used as they are, never refitted.
        learned_on_all = []
        for schema, processors in (
            (self.input_schema, self._input_processors),
            (self.output_schema, self._output_processors),
        ):
            for key, spec in schema.items():
                if key not in processors:
                    processor = self._get_processor_instance(spec)
                    processor.fit(fit_samples, key)
                    processors[key] = processor
                    if split is None and getattr(processor, "learns_statistics", False):
                        learned_on_all.append(key)
        if learned_on_all:
            logger.warning(
                "Processors for %s learn statistics from the data and were fitted "
                "on all samples, so validation/test patients shaped them. Fit on "
                "the training patients with set_task(task, split=PatientSplit(...)).",
                ", ".join(learned_on_all),
            )

        self._fitted = True

    def transform(self, sample: dict[str, bytes]) -> Dict[str, Any]:
        """Transform a single serialized (pickled) sample using fitted processors.

        Args:
            sample: A mapping with a single key `"sample"` whose value is a
                pickled Python dictionary (produced by `pickle.dumps`). The
                pickled dictionary should mirror the schema that was used to
                fit this builder.

        Returns:
            A Python dictionary where each key is either an input or output
            feature name. Values for keys present in the corresponding fitted
            processors have been processed through their FeatureProcessor and
            are returned as the output of that processor. Keys not covered by
            the input/output processors are returned unchanged.
        """
        if not self._fitted:
            raise RuntimeError("SampleBuilder.fit must be called before transform().")

        transformed: Dict[str, Any] = {}
        for key, value in pickle.loads(sample["sample"]).items():
            if key in self._input_processors:
                # Skip ignored features
                if isinstance(self._input_processors[key], IgnoreProcessor):
                    continue
                transformed[key] = self._input_processors[key].process(value)
            elif key in self._output_processors:
                # Skip ignored features
                if isinstance(self._output_processors[key], IgnoreProcessor):
                    continue
                transformed[key] = self._output_processors[key].process(value)
            else:
                transformed[key] = value
        return transformed

    def encode(self, transformed: dict[str, Any]) -> dict[str, Any]:
        """Lays out one processed sample for the disk (litdata) cache.

        litdata infers one serializer per value from the first sample it
        writes and expects every later sample to have the same structure, so a
        list whose length varies between samples cannot be written as-is.
        Fields whose processor declares ``stores_tensor`` stay tensors; every
        other field (including keys not in the schema, such as ``patient_id``)
        goes into one pickled payload, which has the same structure in every
        sample whatever the values look like. :class:`SampleDataset` decodes it
        on read, so callers get back the original Python objects.

        Args:
            transformed: Output of :meth:`transform`.

        Returns:
            A record with the tensor fields plus one bytes payload.

        Raises:
            ValueError: If a field declared ``stores_tensor`` is not a tensor.
        """
        tensor_fields = self._tensor_fields()
        record: dict[str, Any] = {}
        objects: dict[str, Any] = {}
        for key, value in transformed.items():
            if key in tensor_fields:
                if not isinstance(value, torch.Tensor):
                    processor = self._input_processors.get(key) or self._output_processors.get(key)
                    raise ValueError(
                        f"Field '{key}': {type(processor).__name__} declares "
                        f"stores_tensor=True but process() returned "
                        f"{type(value).__name__}. Set stores_tensor = False on the "
                        "processor, or return a torch.Tensor."
                    )
                record[key] = value
            else:
                objects[key] = value
        record[_OBJECTS_KEY] = pickle.dumps(
            {"order": list(transformed), "values": objects}, protocol=_PICKLE_PROTOCOL
        )
        return record

    def transform_for_storage(self, sample: dict[str, bytes]) -> dict[str, Any]:
        """:meth:`transform` followed by :meth:`encode`, for writing to disk."""
        return self.encode(self.transform(sample))

    def _tensor_fields(self) -> set:
        processors = {**self._input_processors, **self._output_processors}
        return {
            key
            for key, processor in processors.items()
            if getattr(processor, "stores_tensor", False)
        }

    def save(self, path: str) -> None:
        """Save fitted metadata to the given path as a pickled file.

        Args:
            path: Location where the builder will write a pickled metadata file
                (commonly named `schema.pkl`). The saved metadata contains
                the fitted input/output schemas, processors, and index
                mappings. This file is read by `SampleDataset` during
                construction.
        """
        if not self._fitted:
            raise RuntimeError("SampleBuilder.fit must be called before save().")
        metadata = {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "input_processors": self._input_processors,
            "output_processors": self._output_processors,
            "patient_to_index": self._patient_to_index,
            "record_to_index": self._record_to_index,
            "fit_split": self._split.to_dict() if self._split is not None else None,
            "storage_format": STORAGE_FORMAT,
        }
        with open(path, "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load(path: str) -> "SampleBuilder":
        """Load a SampleBuilder from a pickled metadata file.

        Args:
            path: Location of the pickled metadata file (commonly named `schema.pkl`).

        Returns:
            A SampleBuilder instance with loaded metadata.
        """
        with open(path, "rb") as f:
            metadata = pickle.load(f)

        builder = SampleBuilder(
            input_schema=metadata["input_schema"],
            output_schema=metadata["output_schema"],
        )
        builder._input_processors = metadata["input_processors"]
        builder._output_processors = metadata["output_processors"]
        builder._patient_to_index = metadata["patient_to_index"]
        builder._record_to_index = metadata["record_to_index"]
        builder._fitted = True
        return builder


class SampleDataset(litdata.StreamingDataset):
    """A streaming dataset that loads sample metadata and processors from disk.

    SampleDataset expects the `path` directory to contain a `schema.pkl`
    file created by a `SampleBuilder.save(...)` call. The `schema.pkl` must
    include the fitted `input_schema`, `output_schema`, `input_processors`,
    `output_processors`, `patient_to_index` and `record_to_index` mappings.

    Attributes:
        input_schema: The configuration used to instantiate processors for
            input features (string aliases or processor specs).
        output_schema: The configuration used to instantiate processors for
            output features.
        input_processors: A mapping of input feature names to fitted
            FeatureProcessor instances.
        output_processors: A mapping of output feature names to fitted
            FeatureProcessor instances.
        patient_to_index: Dictionary mapping patient IDs to the list of
            sample indices associated with that patient.
        record_to_index: Dictionary mapping record/visit IDs to the list of
            sample indices associated with that record.
        dataset_name: Optional human friendly dataset name.
        task_name: Optional human friendly task name.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(  # doctest: +SKIP
        ...     samples=[
        ...         {"patient_id": "p1", "feature": 1, "label": 0},
        ...         {"patient_id": "p2", "feature": 2, "label": 1},
        ...     ],
        ...     input_schema={"feature": "raw"},
        ...     output_schema={"label": "raw"},
        ...     in_memory=False,
        ... )
        >>> dataset.subset([1]).patient_to_index  # doctest: +SKIP
        {'p2': [0]}
        >>> dataset.close()  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str,
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize a SampleDataset pointing at a directory created by SampleBuilder.

        Args:
            path: Path to a directory containing a `schema.pkl` produced by
                `SampleBuilder.save` and associated pickled sample files.
            dataset_name: Optional human-friendly dataset name.
            task_name: Optional human-friendly task name.
            **kwargs: Extra keyword arguments forwarded to
                `litdata.StreamingDataset` (such as streaming options).
        """
        super().__init__(path, **kwargs)

        self.path = path
        self.dataset_name = "" if dataset_name is None else dataset_name
        self.task_name = "" if task_name is None else task_name

        with open(f"{path}/schema.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.input_schema: dict[str, Any] = metadata["input_schema"]
        self.output_schema: dict[str, Any] = metadata["output_schema"]
        self.input_processors: dict[str, FeatureProcessor] = metadata["input_processors"]
        self.output_processors: dict[str, FeatureProcessor]= metadata["output_processors"]
        self._remove_ignored_processors()

        self.patient_to_index = metadata["patient_to_index"]
        self.record_to_index = metadata["record_to_index"]
        # The split the processors were fitted on, or None if fitted on all
        # samples (also None for caches written before this was recorded).
        self.fit_split: dict | None = metadata.get("fit_split")
        self.storage_format: int = metadata.get("storage_format", 1)

    def __getitem__(self, index: Any) -> Any:
        """Returns the sample(s) at ``index`` as the original Python objects.

        Index access, iteration, ``subset()`` views and ``set_task`` split parts
        all read through here, so every path decodes the stored payload.
        """
        return _decode_record(super().__getitem__(index))

    def _remove_ignored_processors(self):
        """Remove any processors that are IgnoreProcessor instances."""
        for key in [
            key
            for key, proc in self.input_processors.items()
            if isinstance(proc, IgnoreProcessor)
        ]:
            del self.input_processors[key]
            del self.input_schema[key]

        for key in [
            key
            for key, proc in self.output_processors.items()
            if isinstance(proc, IgnoreProcessor)
        ]:
            del self.output_processors[key]
            del self.output_schema[key]

    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            str: A string with the dataset and task names.
        """
        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def subset(self, indices: Union[Sequence[int], slice]) -> "SampleDataset":
        """Create a StreamingDataset restricted to the provided indices."""

        new_dataset = deepcopy_dataset(self)

        if len(new_dataset.subsampled_files) != len(new_dataset.region_of_interest):
            raise ValueError(
                "The provided dataset has mismatched subsampled_files and region_of_interest lengths."
            )

        dataset_length = sum(
            end - start for start, end in new_dataset.region_of_interest
        )

        if isinstance(indices, slice):
            indices = range(*indices.indices(dataset_length))

        indices = list(indices)

        if any(idx < 0 or idx >= dataset_length for idx in indices):
            raise ValueError(
                f"Subset indices must be in [0, {dataset_length - 1}] for the provided dataset."
            )

        # Build chunk boundaries so we can translate global indices into
        # chunk-local (start, end) pairs that litdata understands.
        chunk_starts: List[int] = []
        chunk_boundaries: List[Tuple[str, int, int, int, int]] = []
        cursor = 0
        for filename, (roi_start, roi_end) in zip(
            new_dataset.subsampled_files, new_dataset.region_of_interest
        ):
            chunk_len = roi_end - roi_start
            if chunk_len <= 0:
                continue
            chunk_starts.append(cursor)
            chunk_boundaries.append(
                (filename, roi_start, roi_end, cursor, cursor + chunk_len)
            )
            cursor += chunk_len

        new_subsampled_files: List[str] = []
        new_roi: List[Tuple[int, int]] = []
        prev_chunk_idx: Optional[int] = None

        for idx in indices:
            chunk_idx = bisect_right(chunk_starts, idx) - 1
            if chunk_idx < 0 or idx >= chunk_boundaries[chunk_idx][4]:
                raise ValueError(f"Index {idx} is out of bounds for the dataset.")

            filename, roi_start, _, global_start, _ = chunk_boundaries[chunk_idx]
            offset_in_chunk = roi_start + (idx - global_start)

            if (
                new_roi
                and prev_chunk_idx == chunk_idx
                and offset_in_chunk == new_roi[-1][1]
            ):
                new_roi[-1] = (new_roi[-1][0], new_roi[-1][1] + 1)
            else:
                new_subsampled_files.append(filename)
                new_roi.append((offset_in_chunk, offset_in_chunk + 1))

            prev_chunk_idx = chunk_idx

        new_dataset.subsampled_files = new_subsampled_files
        new_dataset.region_of_interest = new_roi
        new_dataset.patient_to_index = _remap_index_mapping(
            self.patient_to_index, indices
        )
        new_dataset.record_to_index = _remap_index_mapping(
            self.record_to_index, indices
        )
        new_dataset.reset()

        return new_dataset

    def close(self) -> None:
        """Cleans up any temporary directories used by the dataset."""
        if self.input_dir.path is not None and Path(self.input_dir.path).exists():
            shutil.rmtree(self.input_dir.path)

    # --------------------------------------------------------------
    # Context manager support
    # --------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class InMemorySampleDataset(SampleDataset):
    """A SampleDataset that loads all samples into memory for fast access.

    InMemorySampleDataset extends SampleDataset by eagerly loading and
    transforming all samples into memory during initialization. This allows
    for fast, repeated access to samples without disk I/O, at the cost of
    higher memory usage.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[{"patient_id": "p1", "feature": 1, "label": 0}],
        ...     input_schema={"feature": "raw"},
        ...     output_schema={"label": "raw"},
        ... )
        >>> dataset[0]["patient_id"]
        'p1'

    Note:
        This class is intended for testing and debugging purposes where
        dataset sizes are small enough to fit into memory.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> None:
        """Initialize an InMemorySampleDataset from in-memory samples.

        This constructor fits a SampleBuilder on the provided samples,
        transforms all samples into memory, and sets up the dataset attributes.

        Args:
            samples: A list of sample dictionaries (in-memory).
            input_schema: Schema describing how input keys should be handled.
            output_schema: Schema describing how output keys should be handled.
            dataset_name: Optional human-friendly dataset name.
            task_name: Optional human-friendly task name.
            input_processors: Optional pre-fitted input processors to use instead
                of creating new ones from the input_schema.
            output_processors: Optional pre-fitted output processors to use
                instead of creating new ones from the output_schema.
        """
        builder = SampleBuilder(
            input_schema=input_schema,
            output_schema=output_schema,
            input_processors=input_processors,
            output_processors=output_processors,
        )
        builder.fit(samples)

        self.dataset_name = "" if dataset_name is None else dataset_name
        self.task_name = "" if task_name is None else task_name

        self.input_schema = builder.input_schema
        self.output_schema = builder.output_schema
        self.input_processors = builder.input_processors
        self.output_processors = builder.output_processors
        self._remove_ignored_processors()

        self.patient_to_index = builder.patient_to_index
        self.record_to_index = builder.record_to_index

        self._data = [builder.transform({"sample": pickle.dumps(s)}) for s in samples]

        self._shuffle = False

    def set_shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # type: ignore
        """Retrieve a processed sample by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A dictionary containing processed input and output features.
        """
        return self._data[index]

    def __iter__(self) -> Iterable[Dict[str, Any]]:  # type: ignore
        """Returns an iterator over all samples in the dataset.

        Returns:
            An iterator yielding processed sample dictionaries.
        """
        if self._shuffle:
            shuffled_data = self._data[:]
            random.shuffle(shuffled_data)
            return iter(shuffled_data)
        else:
            return iter(self._data)

    def subset(self, indices: Union[Sequence[int], slice]) -> SampleDataset:
        if isinstance(indices, slice):
            subset_indices = list(range(*indices.indices(len(self))))
            samples = self._data[indices]
        else:
            raw_indices = list(indices)
            samples = [self._data[i] for i in raw_indices]
            subset_indices = [
                index if index >= 0 else len(self) + index for index in raw_indices
            ]

        new_dataset = copy.deepcopy(self)
        new_dataset._data = samples
        new_dataset.patient_to_index = _remap_index_mapping(
            self.patient_to_index, subset_indices
        )
        new_dataset.record_to_index = _remap_index_mapping(
            self.record_to_index, subset_indices
        )
        return new_dataset

    def close(self) -> None:
        pass  # No temporary directories to clean up for in-memory dataset


def create_sample_dataset(
    samples: List[Dict[str, Any]],
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    dataset_name: Optional[str] = None,
    task_name: Optional[str] = None,
    input_processors: Optional[Dict[str, FeatureProcessor]] = None,
    output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    in_memory: bool = True,
):
    """Convenience helper to create an on-disk SampleDataset from in-memory samples.

    This helper will:
      - Create a temporary directory for the dataset output.
      - Fit a `SampleBuilder` with the provided schemas and samples.
      - Save the fitted `schema.pkl` to the temporary directory.
      - Use `litdata.optimize` with `builder.transform` to write serialized
        and chunked sample files into the directory.
      - Return a `SampleDataset` instance pointed at the temporary directory.

    Args:
        samples: A list of Python dictionaries representing raw samples.
        input_schema: Schema describing how input keys should be handled.
        output_schema: Schema describing how output keys should be handled.
        dataset_name: Optional dataset name to attach to the returned
            SampleDataset instance.
        task_name: Optional task name to attach to the returned SampleDataset
            instance.
        input_processors: Optional pre-fitted input processors to use instead
            of creating new ones from the input_schema.
        output_processors: Optional pre-fitted output processors to use
            instead of creating new ones from the output_schema.
        in_memory: If True, returns an InMemorySampleDataset instead of
            a disk-backed SampleDataset.

    Returns:
        An instance of `SampleDataset` loaded from the temporary directory
        containing the optimized, chunked samples and `schema.pkl` metadata.

    Non-tensor fields (raw values, keys outside the schema) may vary in length
    and shape between samples; the disk-backed dataset returns them unchanged.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "codes": ["A", "B"], "times": [0.5, 1.0], "label": 1},
        ...     {"patient_id": "p2", "codes": ["C"], "times": [0.2], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"codes": "sequence", "times": "raw"}, {"label": "binary"}
        ... )
        >>> dataset[1]["times"]
        [0.2]
    """
    if in_memory:
        return InMemorySampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name=dataset_name,
            task_name=task_name,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        path = Path(tempfile.mkdtemp())

        builder = SampleBuilder(
            input_schema=input_schema,  # type: ignore
            output_schema=output_schema,  # type: ignore
            input_processors=input_processors,
            output_processors=output_processors,
        )
        builder.fit(samples)
        builder.save(str(path / "schema.pkl"))
        litdata.optimize(
            fn=builder.transform_for_storage,
            inputs=[{"sample": pickle.dumps(x)} for x in samples],
            output_dir=str(path),
            chunk_bytes="64MB",
            num_workers=0,
        )

        return SampleDataset(
            path=str(path),
            dataset_name=dataset_name,
            task_name=task_name,
        )
