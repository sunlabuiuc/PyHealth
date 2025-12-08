from collections.abc import Sequence
from pathlib import Path
import pickle
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type
import inspect
import random
from bisect import bisect_right
import litdata
from litdata.utilities.train_test_split import deepcopy_dataset
import copy

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor


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
            input_processors if input_processors is not None else {}
        )
        self._output_processors = (
            output_processors if output_processors is not None else {}
        )
        self._patient_to_index: Dict[str, List[int]] = {}
        self._record_to_index: Dict[str, List[int]] = {}
        self._fitted = False

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
    ) -> None:
        """Fit processors and build mapping indices from an iterator of samples.

        Args:
            samples: Iterable of sample dictionaries (e.g., python dicts). Each
                sample should contain keys covering both the configured
                `input_schema` and `output_schema`. These samples are not
                required to be pickled; `fit` operates on in-memory dicts.

        Behavior:
            - Validates the samples contain all keys specified by the input
              and output schemas.
            - Builds `patient_to_index` and `record_to_index` mappings by
              recording the sample indices associated with `patient_id` and
              `record_id`/`visit_id` fields.
            - Instantiates and fits input/output processors from the provided
              schemas (unless pre-fitted processors were supplied to the
              constructor).
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
        for i, sample in enumerate(samples):
            patient_id = sample.get("patient_id")
            if patient_id is not None:
                self._patient_to_index.setdefault(patient_id, []).append(i)
            record_id = sample.get("record_id", sample.get("visit_id"))
            if record_id is not None:
                self._record_to_index.setdefault(record_id, []).append(i)

        # Fit processors if they were not provided
        if not self._input_processors:
            for key, spec in self.input_schema.items():
                processor = self._get_processor_instance(spec)
                processor.fit(samples, key)
                self._input_processors[key] = processor
        if not self._output_processors:
            for key, spec in self.output_schema.items():
                processor = self._get_processor_instance(spec)
                processor.fit(samples, key)
                self._output_processors[key] = processor

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
                transformed[key] = self._input_processors[key].process(value)
            elif key in self._output_processors:
                transformed[key] = self._output_processors[key].process(value)
            else:
                transformed[key] = value
        return transformed

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
        }
        with open(path, "wb") as f:
            pickle.dump(metadata, f)


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

        self.dataset_name = "" if dataset_name is None else dataset_name
        self.task_name = "" if task_name is None else task_name

        with open(f"{path}/schema.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.input_schema = metadata["input_schema"]
        self.output_schema = metadata["output_schema"]
        self.input_processors = metadata["input_processors"]
        self.output_processors = metadata["output_processors"]

        self.patient_to_index = metadata["patient_to_index"]
        self.record_to_index = metadata["record_to_index"]

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
        new_dataset.reset()

        return new_dataset


class InMemorySampleDataset(SampleDataset):
    """A SampleDataset that loads all samples into memory for fast access.

    InMemorySampleDataset extends SampleDataset by eagerly loading and
    transforming all samples into memory during initialization. This allows
    for fast, repeated access to samples without disk I/O, at the cost of
    higher memory usage.

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
            samples = self._data[indices]
        else:
            samples = [self._data[i] for i in indices]

        new_dataset = copy.deepcopy(self)
        new_dataset._data = samples
        return new_dataset


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
            fn=builder.transform,
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
