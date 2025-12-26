import os

import numpy as np

from pyhealth.datasets import BaseSignalDataset

from pyhealth.datasets.utils import read_edf_data, save_to_npz
from tqdm import tqdm

class SHHSDataset(BaseSignalDataset):
    """EEG and ECG dataset for Sleep Heart Health Study (SHHS)

    Dataset is available at https://sleepdata.org/datasets/shhs

    The Sleep Heart Health Study (SHHS) is a multi-center cohort study implemented by the National Heart Lung & Blood Institute to determine the cardiovascular and other consequences of sleep-disordered breathing. It tests whether sleep-related breathing is associated with an increased risk of coronary heart disease, stroke, all cause mortality, and hypertension.  In all, 6,441 men and women aged 40 years and older were enrolled between November 1, 1995 and January 31, 1998 to take part in SHHS Visit 1. During exam cycle 3 (January 2001- June 2003), a second polysomnogram (SHHS Visit 2) was obtained in 3,295 of the participants. CVD Outcomes data were monitored and adjudicated by parent cohorts between baseline and 2011. More than 130 manuscripts have been published investigating predictors and outcomes of sleep disorders.

    This dataset supports both EEG and ECG signal processing.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain EDF files and annotations).
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "sleep staging", "ecg analysis").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.
        patients: Dict[str, List[Dict]], processed patient data with EEG/ECG file paths.

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> dataset = SHHSDataset(
        ...         root="/srv/local/data/SHHS/",
        ...     )
        >>> # Process ECG data
        >>> dataset.process_ECG_data(out_dir="./ecg_output")
    """

    def __init__(self, root, dev=False, refresh_cache=False, **kwargs):
        """Initialize SHHS Dataset"""
        super().__init__()
        self.root = root
        self.dev = dev
        self.refresh_cache = refresh_cache
        self.filepath = os.path.join(os.path.expanduser("~"), ".cache", "pyhealth_shhs")
        self.patients = self.process_EEG_data()

    def parse_patient_id(self, file_name):
        """
        Args:
            file_name: the file name of the shhs datasets. e.g., shhs1-200001.edf
        Returns:
            patient_id: the patient id of the shhs datasets. e.g., 200001
        """
        return file_name.split("-")[1].split(".")[0]

    def process_EEG_data(self):

        # get shhs1
        shhs1 = []
        if os.path.exists(os.path.join(self.root, "edfs/shhs1")):
            print("shhs1 exists and load shhs1")
            shhs1 = os.listdir(os.path.join(self.root, "edfs/shhs1"))
        else:
            print("shhs1 does not exist")

        # get shhs2
        shhs2 = []
        if os.path.exists(os.path.join(self.root, "edfs/shhs2")):
            print("shhs2 exists and load shhs2")
            shhs2 = os.listdir(os.path.join(self.root, "edfs/shhs2"))
        else:
            print("shhs2 does not exist")

        # get all patient ids
        patient_ids = np.unique([self.parse_patient_id(file) for file in shhs1 + shhs2])
        if self.dev:
            patient_ids = patient_ids[:5]
        # get patient to record maps
        #    - key: pid:
        #    - value: [{"load_from_path": None, "file": None, "save_to_path": None}, ...]
        patients = {pid: [] for pid in patient_ids}

        # parse shhs1
        for file in shhs1:
            pid = self.parse_patient_id(file)
            if pid in patient_ids:
                patients[pid].append(
                    {
                        "load_from_path": self.root,
                        "signal_file": os.path.join("edfs/shhs1", file),
                        "label_file": os.path.join("annotations-events-profusion/shhs1", f"shhs1-{pid}-profusion.xml"),
                        "save_to_path": os.path.join(self.filepath),
                    }
                )

        # parse shhs2
        for file in shhs2:
            pid = self.parse_patient_id(file)
            if pid in patient_ids:
                patients[pid].append(
                    {
                        "load_from_path": self.root,
                        "signal_file": os.path.join("edfs/shhs2", file),
                        "label_file": os.path.join("annotations-events-profusion/label", f"shhs2-{pid}-profusion.xml"),
                        "save_to_path": os.path.join(self.filepath),
                    }
                )
        return patients

    def process_ECG_data(self, out_dir, target_fs=None, select_chs=["ECG"], require_annotations=False):
        """
        Extract SHHS ECG signals + labels and save them as .npz files.

        Args:
            out_dir: Destination directory for generated .npz files.
            target_fs: Optional int, target sampling rate (e.g., 100 Hz).
            select_chs: list of channels to extract, default ECG.
            require_annotations: If True, skip files without annotations. If False, process signals without labels.

        Expected SHHS directory structure:
            root/
                edfs/shhs1/*.edf
                edfs/shhs2/*.edf
                annotations-events-profusion/shhs1/*.xml
                annotations-events-profusion/label/*.xml (for shhs2)
        """

        shhs_configs = [
            {
                "data_dir": os.path.join(self.root, "edfs", "shhs1"),
                "annotation_dir": os.path.join(self.root, "annotations-events-profusion", "shhs1"),
                "label": "shhs1"
            },
            {
                "data_dir": os.path.join(self.root, "edfs", "shhs2"),
                "annotation_dir": os.path.join(self.root, "annotations-events-profusion", "label"),
                "label": "shhs2"
            }
        ]

        os.makedirs(out_dir, exist_ok=True)
        processed_count = 0
        skipped_count = 0

        for config in shhs_configs:
            data_dir = config["data_dir"]
            annotation_dir = config["annotation_dir"]
            dir_label = config["label"]

            if not os.path.exists(data_dir):
                print(f"Directory missing: {data_dir}")
                continue

            files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
            print(f"Processing ECG for {dir_label}: {len(files)} EDF files found")

            if not files:
                continue

            for file in tqdm(files, desc=f"Processing {dir_label}"):
                sid = self.parse_patient_id(file)
                data_path = os.path.join(data_dir, file)

                # Determine annotation file path
                if dir_label == "shhs1":
                    annotation_filename = f"shhs1-{sid}-profusion.xml"
                else:  # shhs2
                    annotation_filename = f"shhs2-{sid}-profusion.xml"
                
                label_path = os.path.join(annotation_dir, annotation_filename)

                # Check if annotation exists
                has_annotation = os.path.exists(label_path)
                
                if require_annotations and not has_annotation:
                    print(f"Skipping {sid}: missing annotation {label_path}")
                    skipped_count += 1
                    continue

                try:
                    if has_annotation:
                        # Process with annotations
                        data, fs, stages = read_edf_data(
                            data_path=data_path,
                            label_path=label_path,
                            dataset="SHHS",
                            select_chs=select_chs,
                            target_fs=target_fs,
                        )
                        outfile = os.path.join(out_dir, f"{dir_label}-{sid}.npz")
                        save_to_npz(outfile, data, stages, fs)
                        print(f"✓ Processed {sid} with annotations")
                    else:
                        # Process without annotations (signals only) - skip label_path entirely
                        try:
                            # Try to read EDF file directly without using read_edf_data for labels
                            import mne
                            raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
                            
                            # Select channels
                            if select_chs:
                                available_chs = [ch for ch in select_chs if ch in raw.ch_names]
                                if not available_chs:
                                    print(f"⚠ No requested channels found in {sid}. Available: {raw.ch_names}")
                                    skipped_count += 1
                                    continue
                                raw = raw.pick_channels(available_chs)
                            
                            # Get data and sampling frequency
                            data = raw.get_data()
                            fs = raw.info['sfreq']
                            
                            # Resample if needed
                            if target_fs and target_fs != fs:
                                raw = raw.resample(target_fs)
                                data = raw.get_data()
                                fs = target_fs
                            
                            outfile = os.path.join(out_dir, f"{dir_label}-{sid}_no_labels.npz")
                            save_to_npz(outfile, data, None, fs)
                            print(f"⚠ Processed {sid} without annotations (signals only)")
                        
                        except Exception as edf_error:
                            print(f"❌ Error reading EDF file for {sid}: {edf_error}")
                            skipped_count += 1
                            continue
                    
                    processed_count += 1

                except Exception as e:
                    print(f"❌ Error processing patient {sid}: {e}")
                    skipped_count += 1

        print(f"\nECG extraction completed:")
        print(f"  ✓ Successfully processed: {processed_count} files")
        print(f"  ⚠ Skipped/failed: {skipped_count} files")
        
        return processed_count > 0

if __name__ == "__main__":
    dataset = SHHSDataset(
        root="/srv/local/data/SHHS/polysomnography",
        dev=True,
        refresh_cache=True,
    )
    print(f"Dataset loaded with {len(dataset.patients)} patients")
    print(list(dataset.patients.items())[0])
    
    # Example ECG processing
    # dataset.process_ECG_data(out_dir="./ecg_output")
