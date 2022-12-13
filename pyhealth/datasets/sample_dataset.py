from collections import Counter
from typing import Dict, List

from torch.utils.data import Dataset

from pyhealth.datasets.utils import list_nested_levels, flatten_list


class SampleDataset(Dataset):
    """Sample dataset class.

    This class the takes a list of samples as input (either from
    `BaseDataset.set_task()` or user-provided input), and provides
    a uniform interface for accessing the samples.

    Args:
        samples: a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
        dataset_name: the name of the dataset. Default is None.
        task_name: the name of the task. Default is None.

    Currently, the following types of attributes are supported:
        - a single value. Type: int/float/str. Dim: 0.
        - a single vector. Type: int/float. Dim: 1.
        - a list of codes. Type: str. Dim: 2.
        - a list of vectors. Type: int/float. Dim: 2.
        - a list of list of codes. Type: str. Dim: 3.
        - a list of list of vectors. Type: int/float. Dim: 3.

    Attributes:
        input_info: Dict, a dict whose keys are the same as the keys in the
            samples, and values are the corresponding input information:
            - "type": the element type of each key attribute, one of float, int, str.
            - "dim": the list dimension of each key attribute, one of 0, 1, 2, 3.
            - "len": the length of the vector, only valid for vector-based attributes.
        patient_to_index: Dict[str, List[int]], a dict mapping patient_id to
            a list of sample indices.
        visit_to_index: Dict[str, List[int]], a dict mapping visit_id to a list
            of sample indices.
    """

    def __init__(self, samples: List[Dict], dataset_name="", task_name=""):
        self.samples = samples
        self.dataset_name: str = dataset_name
        self.task_name: str = task_name
        self.input_info: Dict = self._validate()
        self.patient_to_index: Dict[str, List[int]] = self._index_patient()
        self.visit_to_index: Dict[str, List[int]] = self._index_visit()

    def _validate(self) -> Dict:
        """Helper function which validates the samples.

        Will be called in `self.__init__()`.

        Returns:
            input_info: Dict, a dict whose keys are the same as the keys in the
                samples, and values are the corresponding input information:
                - "type": the element type of each key attribute, one of float,
                    int, str.
                - "dim": the list dimension of each key attribute, one of 0, 1, 2, 3.
                - "len": the length of the vector, only valid for vector-based
                    attributes.
        """
        """ 1. Check if all samples are of type dict. """
        assert all(
            [isinstance(s, dict) for s in self.samples],
        ), "Each sample should be a dict"
        keys = self.samples[0].keys()

        """ 2. Check if all samples have the same keys. """
        assert all(
            [set(s.keys()) == set(keys) for s in self.samples]
        ), "All samples should have the same keys"

        """ 3. Check if "patient_id" and "visit_id" are in the keys."""
        assert "patient_id" in keys, "patient_id should be in the keys"
        assert "visit_id" in keys, "visit_id should be in the keys"

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float 
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            """ 
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples])
            assert (
                    len(levels) == 1 and len(list(levels)[0]) == 1
            ), f"Key {key} has mixed nested list levels across samples"
            level = levels.pop()[0]
            assert level in [
                0,
                1,
                2,
                3,
            ], f"Key {key} has unsupported nested list level across samples"

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [
                    i for s in self.samples for i in s[key]
                ]
            elif level == 2:
                flattened_values = [
                    j for s in self.samples for i in s[key] for j in i
                ]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float, 
            int, or str.
            """
            types = set([type(v) for v in flattened_values])
            assert len(types) == 1, f"Key {key} has mixed types across samples"
            type_ = types.pop()
            assert type_ in [
                float,
                int,
                str,
            ], f"Key {key} has unsupported type across samples"

            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                assert type_ in [
                    float,
                    int,
                    str,
                ], f"Key {key} has unsupported type across samples"
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                assert type_ in [
                    float,
                    int,
                ], f"Key {key} has unsupported type across samples"
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def _index_patient(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by patient_id.

        Will be called in `self.__init__()`.
        Returns:
            patient_to_index: Dict[str, int], a dict mapping patient_id to a list
                of sample indices.
        """
        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def _index_visit(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by visit_id.

        Will be called in `self.__init__()`.

        Returns:
            visit_to_index: Dict[str, int], a dict mapping visit_id to a list
                of sample indices.
        """
        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def get_all_tokens(
            self, key: str, remove_duplicates: bool = True, sort: bool = True
    ) -> List[str]:
        """Gets all tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.
            remove_duplicates: whether to remove duplicates. Default is True.
            sort: whether to sort the tokens by alphabet order. Default is True.

        Returns:
            tokens: a list of tokens.
        """
        input_type = self.input_info[key]["type"]
        input_dim = self.input_info[key]["dim"]
        if input_type in [float, int]:
            assert input_dim == 0, f"Cannot get tokens for vector with key {key}"

        tokens = []
        for sample in self.samples:
            if input_dim == 0:
                # a single value
                tokens.append(sample[key])
            elif input_dim == 2:
                # a list of codes
                tokens.extend(sample[key])
            elif input_dim == 3:
                # a list of list of codes
                tokens.extend(flatten_list(sample[key]))
            else:
                raise NotImplementedError
        if remove_duplicates:
            tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        """Gets the distribution of tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.

        Returns:
            distribution: a dict mapping token to count.
        """

        tokens = self.get_all_tokens(key, remove_duplicates=False, sort=False)
        counter = Counter(tokens)
        return counter

    def __getitem__(self, index) -> Dict:
        """Returns a sample by index.

        Returns:
             Dict, a dict with patient_id, visit_id, and other task-specific
                attributes as key. Conversion to index/tensor will be done
                in the model.
        """
        return self.samples[index]

    def __str__(self):
        """Prints some information of the dataset."""
        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def stat(self) -> str:
        """Returns some statistics of the task-specific dataset."""
        lines = list()
        lines.append(f"Statistics of sample dataset:")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Task: {self.task_name}")
        lines.append(f"\t- Number of samples: {len(self)}")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        lines.append(f"\t- Number of patients: {num_patients}")
        num_visits = len(set([sample["visit_id"] for sample in self.samples]))
        lines.append(f"\t- Number of visits: {num_visits}")
        lines.append(
            f"\t- Number of visits per patient: {len(self) / num_patients:.4f}"
        )
        for key in self.samples[0]:
            if key in ["patient_id", "visit_id"]:
                continue
            input_type = self.input_info[key]["type"]
            input_dim = self.input_info[key]["dim"]

            if input_dim <= 1:
                # a single value or vector
                num_events = [1 for sample in self.samples]
            elif input_dim == 2:
                # a list
                num_events = [len(sample[key]) for sample in self.samples]
            elif input_dim == 3:
                # a list of list
                num_events = [
                    len(flatten_list(sample[key])) for sample in self.samples
                ]
            else:
                raise NotImplementedError
            lines.append(f"\t- {key}:")
            lines.append(
                f"\t\t- Number of {key} per sample: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
            if input_type == str or input_dim == 0:
                # single value or code-based
                lines.append(
                    f"\t\t- Number of unique {key}: {len(self.get_all_tokens(key))}"
                )
                distribution = self.get_distribution_tokens(key)
                top10 = sorted(
                    distribution.items(), key=lambda x: x[1], reverse=True
                )[:10]
                lines.append(f"\t\t- Distribution of {key} (Top-10): {top10}")
            else:
                # vector-based
                vector = self.samples[0][key]
                lines.append(f"\t\t- Length of {key}: {self.input_info[key]['len']}")
        print("\n".join(lines))
        return "\n".join(lines)


if __name__ == "__main__":
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0-0",
            "single_vector": [1, 2, 3],
            "list_codes": ["cond-33", "cond-86", "cond-80"],
            "list_vectors": [[1, 2, 3], [4, 5, 6]],
            "list_list_codes": [["cond-33"], ["cond-86", "cond-80"]],
            "list_list_vectors": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1-0",
            "single_vector": [1, 2, 5],
            "list_codes": ["cond-33", "cond-86", "cond-80"],
            "list_vectors": [[1, 3, 3], [4, 5, 1]],
            "list_list_codes": [["cond-33", "cond-86"], ["cond-80"]],
            "list_list_vectors": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 1]]],
            "label": 0,
        },
    ]

    dataset = SampleDataset(samples=samples)

    dataset.stat()
    data = iter(dataset)
    print(next(data))
