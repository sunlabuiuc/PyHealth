"""
PyHealth dataset for the PhysioNet/Computing in Cardiology Challenge 2020.

Dataset link:
    https://physionet.org/content/challenge-2020/1.0.2/

Dataset paper: (please cite if you use this dataset)
    Perez Alday EA, Gu A, Shah AJ, Robichaux C, Wong AI, Liu C, Liu F, 
    Rad AB, Elola A, Seyedi S, Li Q, Sharma A, Clifford GD, Reyna MA. 
    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020. 
    Physiol Meas. 2020 Nov 11. http://doi.org/10.1088/1361-6579/abc960.

Dataset resource:
    Perez Alday, E. A., Gu, A., Shah, A., Liu, C., Sharma, A., Seyedi, S., 
    Bahrami Rad, A., Reyna, M., & Clifford, G. (2022). Classification of 
    12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020 (version 1.0.2). 
    PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dvyd-kd57

PhysioNet:
    Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R.,
    ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: 
    Components of a new research resource for complex physiologic signals. 
    Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

Author:
    John Ma (jm119@illinois.edu)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

SUBDATASET_NAMES: List[str] = [
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ptb",
    "ptb-xl",
    "st_petersburg_incart",
]

class Cardiology2Dataset(BaseDataset):
    """Dataset class for the PhysioNet/CinC Challenge 2020 12-lead ECG data.

    The dataset bundles six sub-collections of 12-lead ECG recordings stored as
    MATLAB '.mat' files with companion '.hea' header files containing
    SNOMED-CT diagnosis codes, patient sex, and patient age.

    Dataset is available at:
        https://physionet.org/content/challenge-2020/1.0.2/

    Args:
        root (str): Root directory of the raw data, e.g.
            '"/data/physionet.org/files/challenge-2020/1.0.2/training"'.
        chosen_dataset (List[int]): Binary list of length 6 indicating which
            sub-datasets to include. Indices correspond to:
            '["cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl", "st_petersburg_incart"]'.
            Default: '[1, 1, 1, 1, 1, 1]' (all six).
        config_path (Optional[str]): Path to the YAML config file. Defaults to
            the bundled 'configs/cardiology.yaml'.

    Attributes:
        classes (List[str]): Union of common SNOMED-CT diagnosis codes across
            five symptom categories (AR, BBBFB, AD, CD, WA).
        chosen_dataset (List[int]): The sub-dataset selection mask.

    Examples:
        >>> from pyhealth.datasets import Cardiology2Dataset
        >>> dataset = Cardiology2Dataset(
        ...     root="/data/physionet.org/files/challenge-2020/1.0.2/training",
        ... )
        >>> dataset.stats()
    """

    """
    Classes:
        Source: https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_scored.csv
    """
    classes: List[str] = [
        "270492004",
        "164889003",
        "164890007",
        "426627000",
        "713427006",
        "713426002",
        "445118002",
        "39732003",
        "164909002",
        "251146004",
        "698252002",
        "10370003",
        "284470004",
        "427172004",
        "164947007",
        "111975006",
        "164917005",
        "47665007",
        "59118001",
        "427393009",
        "426177001",
        "426783006",
        "427084000",
        "63593006",
        "164934002",
        "59931005",
        "17338001",
    ]

    def __init__(
        self,
        root: str,
        chosen_dataset: List[int] = [1, 1, 1, 1, 1, 1],
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "cardiology.yaml"
        ),
        **kwargs,
    ) -> None:
        if len(chosen_dataset) != 6 or not all(v in (0, 1) for v in chosen_dataset):
            raise ValueError(
                "chosen_dataset must be a binary list of length 6, e.g. [1,1,1,1,1,1]"
            )

        self.chosen_dataset = chosen_dataset
        self._index_data(root)
        super().__init__(
            root=root,
            tables=["cardiology"],
            dataset_name="Cardiology",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self):
        """Returns the default multi-label ECG classification task.

        Returns:
            CardiologyMultilabelClassification: the default task.

        Example::
            >>> dataset = Cardiology2Dataset(root="...")
            >>> task = dataset.default_task
        """
        from pyhealth.tasks import CardiologyMultilabelClassification
        return CardiologyMultilabelClassification()

    def _index_data(self, root: str) -> None:
        """Scans all .hea files and writes a flat metadata CSV.

        For each recording the following fields are extracted from the header:

        - patient_id: "{dataset_idx}_{patient_idx}"
        - signal_path: absolute path to the .mat file
        - dx: comma-separated SNOMED-CT diagnosis codes
        - sex: patient sex string (e.g. "Male")
        - age: patient age string (e.g. 63)
        - chosen_dataset: name of the sub-dataset this recording belongs to (e.g "cpsc_2018")

        The resulting table is written to '{root}/cardiology-metadata-pyhealth.csv'

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If 'root' does not exist.
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        out_path = os.path.join(root, "cardiology-metadata-pyhealth.csv")
        if os.path.isfile(out_path):
            logger.info(f"Found existing metadata index: {out_path}")
            logger.info(f"Overwriting existing metadata index...")

        active_datasets = [
            (idx, name)
            for idx, (name, flag) in enumerate(
                zip(SUBDATASET_NAMES, self.chosen_dataset)
            )
            if flag
        ]

        rows = []
        for dataset_idx, name in active_datasets:
            dataset_dir = os.path.join(root, name)
            patient_dirs = sorted(
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            )

            for patient_idx, patient_dir in enumerate(patient_dirs):
                patient_root = os.path.join(dataset_dir, patient_dir)
                pid = f"{dataset_idx}_{patient_idx}"

                for record in self._collect_recordings(patient_root):
                    record["patient_id"] = pid
                    record["chosen_dataset"] = name
                    rows.append(record)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info(
            f"Wrote metadata index with {len(df)} recordings to {out_path}"
        )

    def _collect_recordings(self, patient_dir: str) -> List[Dict]:
        """Collects metadata for all recordings in a patient directory.

        Finds every '.hea' file, checks for a matching '.mat' file,
        and parses the header to extract diagnosis codes and demographics.

        Args:
            patient_dir (str): Absolute path to a patient directory.

        Returns:
            List[Dict]: One dict per valid recording with keys
                'signal_path', 'dx', 'sex', and 'age'.
        """
        records = []
        hea_files = [
            f for f in os.listdir(patient_dir) if f.endswith(".hea")
        ]
        for hea_file in hea_files:
            file_name = hea_file[:-4]
            mat_path = os.path.join(patient_dir, file_name + ".mat")
            hea_path = os.path.join(patient_dir, hea_file)

            if not os.path.isfile(mat_path):
                logger.debug(f"No matching .mat for {hea_path}, skipping")
                continue

            dx, sex, age = self._parse_header(hea_path)
            records.append({
                "signal_path": mat_path,
                "dx": dx,
                "sex": sex,
                "age": age,
            })
        return records

    @staticmethod
    def _parse_header(hea_path: str):
        """Parses Dx, Sex, and Age from a PhysioNet 2020 .hea header file.

        The last few lines of each '.hea' file follow this format:

        # Age: 63
        # Sex: Male
        # Dx: 426783006,164934002

        Args:
            hea_path (str): Path to the '.hea' file.

        Returns:
            Tuple[str, str, str]: '(dx, sex, age)' as raw strings.
                'dx' is a comma-separated SNOMED-CT code string.
        """
        dx = sex = age = ""
        with open(hea_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Dx:"):
                    dx = line.split(":", 1)[1].strip()
                elif line.startswith("# Sex:"):
                    sex = line.split(":", 1)[1].strip()
                elif line.startswith("# Age:"):
                    age = line.split(":", 1)[1].strip()

        return dx, sex, age
