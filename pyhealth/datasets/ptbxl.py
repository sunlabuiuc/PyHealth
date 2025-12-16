"""
PTB-XL Dataset for PyHealth

Based on:
Wagner et al., PTB-XL, a large publicly available electrocardiography dataset.
Scientific Data 7.1 (2020): 154.
"""

import ast
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseDataset):
    """
    Base ECG dataset for PTB-XL

    Dataset is available at https://physionet.org/content/ptb-xl/1.0.3/

    The PTB-XL dataset contains 21,837 clinical 12-lead ECG records from
    18,885 patients of 10 seconds length.

    Args:
        root: root directory of the raw data (should contain ptbxl_database.csv)
        sampling_rate: sampling rate (100 or 500 Hz). Default is 100.
        dataset_name: optional name of the dataset. Default is "ptbxl".
        config_path: optional path to the config file. Default is None (uses built-in config).
        dev: whether to enable dev mode (only use subset). Default is False.

    Attributes:
        sampling_rate: the sampling rate used for loading ECG signals.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> dataset = PTBXLDataset(
        ...     root="/path/to/ptb-xl",
        ...     sampling_rate=100
        ... )
        >>> dataset.stats()
        >>>
        >>> # Get all patient ids
        >>> unique_patients = dataset.unique_patient_ids
        >>> print(f"There are {len(unique_patients)} patients")
        >>>
        >>> # Get single patient data
        >>> patient = dataset.get_patient("1")
        >>> print(f"Patient has {len(patient.data_source)} events")
        >>>
        >>> # Get ECG events
        >>> events = patient.get_events(event_type="ecg")
        >>>
        >>> # Get diagnostic superclass
        >>> diagnostic = events[0].diagnostic_superclass
        >>> print(f"Diagnostic superclass: {diagnostic}")
    """

    def __init__(
        self,
        root: str,
        sampling_rate: int = 100,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
    ):
        if sampling_rate not in [100, 500]:
            raise ValueError("sampling_rate must be 100 or 500")

        self.sampling_rate = sampling_rate

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ptbxl.yaml"

        # Prepare metadata CSV if it doesn't exist
        metadata_path = os.path.join(root, "ptbxl-pyhealth.csv")
        if not os.path.exists(metadata_path):
            logger.info(f"{metadata_path} does not exist, preparing metadata...")
            self._prepare_metadata(root, sampling_rate)

        default_tables = ["ecg"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            dev=dev,
        )

    def _prepare_metadata(self, root: str, sampling_rate: int) -> None:
        """Prepare metadata CSV for PTB-XL dataset.

        This method processes the raw PTB-XL metadata and creates a processed
        CSV file that follows the PyHealth format.

        Args:
            root: Root directory containing the PTB-XL dataset files.
            sampling_rate: The sampling rate to use (100 or 500 Hz).
        """
        # Load database CSV
        csv_path = os.path.join(root, "ptbxl_database.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"ptbxl_database.csv not found at {csv_path}\n"
                "Download PTB-XL from: https://physionet.org/content/ptb-xl/1.0.3/"
            )

        logger.info(f"Loading PTB-XL database from {csv_path}")
        df = pd.read_csv(csv_path, index_col='ecg_id')

        # Load SCP statements for diagnostic labels
        scp_path = os.path.join(root, "scp_statements.csv")
        if not os.path.exists(scp_path):
            raise FileNotFoundError(
                f"scp_statements.csv not found at {scp_path}\n"
                "Download PTB-XL from: https://physionet.org/content/ptb-xl/1.0.3/"
            )

        scp_df = pd.read_csv(scp_path, index_col=0)

        # Parse SCP codes
        df['scp_codes'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Add diagnostic superclass labels
        df = self._add_diagnostic_labels(df, scp_df)

        # Get filename based on sampling rate
        if sampling_rate == 100:
            filename_col = 'filename_lr'  # Low resolution
        else:
            filename_col = 'filename_hr'  # High resolution

        # Create processed dataframe
        records = []
        for ecg_id, row in df.iterrows():
            filename = row[filename_col]
            # Build signal file path (use .dat extension for wfdb format)
            signal_file = os.path.join(root, filename)

            records.append({
                'patient_id': str(row['patient_id']),
                'ecg_id': str(ecg_id),
                'age': row.get('age'),
                'sex': row.get('sex'),
                'height': row.get('height'),
                'weight': row.get('weight'),
                'scp_codes': str(row.get('scp_codes', {})),
                'diagnostic_superclass': str(row.get('diagnostic_superclass', [])),
                'strat_fold': row.get('strat_fold'),
                'signal_file': signal_file,
            })

        processed_df = pd.DataFrame(records)

        # Save processed metadata
        output_path = os.path.join(root, "ptbxl-pyhealth.csv")
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed metadata to {output_path}")

    def _add_diagnostic_labels(self, df: pd.DataFrame, scp_df: pd.DataFrame) -> pd.DataFrame:
        """Add diagnostic superclass labels to the dataframe.

        Args:
            df: The main PTB-XL dataframe.
            scp_df: The SCP statements dataframe.

        Returns:
            The dataframe with diagnostic_superclass column added.
        """
        diagnostic_scp = scp_df[scp_df.diagnostic == 1]

        def aggregate_diagnostic(scp_codes):
            classes = []
            for code in scp_codes.keys():
                if code in diagnostic_scp.index:
                    classes.append(diagnostic_scp.loc[code].diagnostic_class)
            return list(set(classes))

        df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
        return df


if __name__ == "__main__":
    dataset = PTBXLDataset(
        root="/path/to/ptb-xl",
        sampling_rate=100,
        dev=True,
    )
    dataset.stats()
