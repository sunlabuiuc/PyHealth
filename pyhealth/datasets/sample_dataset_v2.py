from typing import Dict, List, Optional

from torch.utils.data import Dataset

import sys
sys.path.append('.')

from pyhealth.datasets.featurizers import ImageFeaturizer, ValueFeaturizer


class SampleDataset(Dataset):
    """Sample dataset class.
    """

    def __init__(
        self,
        samples: List[Dict],
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        if dataset_name is None:
            dataset_name = ""
        if task_name is None:
            task_name = ""
        self.samples = samples
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.transform = None
        # TODO: get rid of input_info
        self.input_info: Dict = self.validate()
        self.build()

    def validate(self):
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for s in self.samples:
            assert input_keys.issubset(s.keys()), \
                "Input schema does not match samples."
            assert output_keys.issubset(s.keys()), \
                "Output schema does not match samples."
        input_info = {}
        # get label signal info
        input_info["label"] = {"type": str, "dim": 0}
        return input_info

    def build(self):
        for k, v in self.input_schema.items():
            if v == "image":
                self.input_schema[k] = ImageFeaturizer()
            else:
                self.input_schema[k] = ValueFeaturizer()
        for k, v in self.output_schema.items():
            if v == "image":
                self.output_schema[k] = ImageFeaturizer()
            else:
                self.output_schema[k] = ValueFeaturizer()
        return

    def __getitem__(self, index) -> Dict:
        """Returns a sample by index.

        Returns:
             Dict, a dict with patient_id, visit_id/record_id, and other task-specific
                attributes as key. Conversion to index/tensor will be done
                in the model.
        """
        out = {}
        for k, v in self.samples[index].items():
            if k in self.input_schema:
                out[k] = self.input_schema[k].encode(v)
            elif k in self.output_schema:
                out[k] = self.output_schema[k].encode(v)
            else:
                out[k] = v

        if self.transform is not None:
            out = self.transform(out)

        return out

    def set_transform(self, transform):
        """Sets the transform for the dataset.

        Args:
            transform: a callable transform function.
        """
        self.transform = transform
        return

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
        # TODO: get rid of this function
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

    def __str__(self):
        """Prints some information of the dataset."""
        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)


if __name__ == "__main__":
    samples = [
        {
            "id": "0",
            "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [
                ["A05B", "A05C", "A06A"],
                ["A11D", "A11E"]
            ],
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "image": "data/COVID-19_Radiography_Dataset/Normal/images/Normal-6335.png",
            "text": "This is a sample text",
            "label": 1,
        },
    ]

    dataset = SampleDataset(
        samples=samples,
        input_schema={
            "id": "str",
            "single_vector": "vector",
            "list_codes": "list",
            "list_vectors": "list",
            "list_list_codes": "list",
            "list_list_vectors": "list",
            "image": "image",
            "text": "text",
        },
        output_schema={
            "label": "label"
        }
    )
    print(dataset[0])
