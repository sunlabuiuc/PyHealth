import subprocess
import shutil
import os
from typing import Optional, List
from urllib.request import urlretrieve

from pyhealth.datasets import BaseSignalDataset

def _download_file(online_filepath, local_filepath, refresh_cache=False):
    if (not os.path.exists(local_filepath)) or refresh_cache:
        urlretrieve(online_filepath, local_filepath)
    return local_filepath

def _unrar_function(rar_path, dst_path):
    if os.name == 'nt':
        try:
            import patoolib
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install patool to download ISRUC data. \
            You might need to have 7z/rar/unrar installed as well.")
        
        patoolib.extract_archive(rar_path, outdir=dst_path)
    else:
        # Linux, we use 7zzs, which can be downloaded from https://www.7-zip.org/download.html
        path_7z = shutil.which('7zzs')
        assert path_7z is not None, "Please download 7z for linux."
        subprocess.call([path_7z, 'x', rar_path, f'-o{dst_path}'])

def _download_ISRUC_group1(data_dir:str, exclude_subjects:List[int]):
    """Download all group 1 data for ISRUC.

    Args:
        data_dir (str): 
            path to download the data.
        exclude_subjects (List[int]): 
            List of subjects to exclude. 
    Returns:
        raw_dir: directory the dataset is extracted to (in data_dir).
    """
    rar_dir = os.path.join(data_dir, 'rar_files')
    raw_dir = os.path.join(data_dir, 'raw')
    for _ in [rar_dir, raw_dir]:
        if not os.path.isdir(_): os.makedirs(_)
    exclude_subjects = set(exclude_subjects)

    for subject_id in range(1, 100 + 1):
        if subject_id in exclude_subjects: continue
        if os.path.isfile(os.path.join(raw_dir, f'{subject_id}/{subject_id}.edf')):
            continue
        rar_url = f"http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/{subject_id}.rar"
        rar_dst = os.path.join(rar_dir, f"{subject_id}.rar")
        _download_file(rar_url, rar_dst)
        _unrar_function(rar_dst, raw_dir)
        os.rename(os.path.join(raw_dir, f'{subject_id}/{subject_id}.rec'),
                  os.path.join(raw_dir, f'{subject_id}/{subject_id}.edf'))
    return raw_dir

class ISRUCDataset(BaseSignalDataset):
    """Base EEG dataset for ISRUC Group I.
    
    Dataset is available at https://sleeptight.isr.uc.pt/

        - The EEG signals are sampled at 200 Hz.
        - There are 100 subjects in the orignal dataset. 
        - Each subject's data is about a night's sleep.


    Args:
        dataset_name: name of the dataset.
            Default is 'ISRUCDataset'.
        root: root directory of the raw data. 
            We expect `root/raw` to contain all extracted files (.txt, .rec, ...)
            You can also download the data to a new directory by using download=True.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: Whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
        download: Whether to download the data automatically. 
            Default is False.
    

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> dataset = ISRUCDataset(
        ...         root="/srv/local/data/data/ISRUC-I",
        ...         download=True,
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """
    _EXCLUDE_SUBJECTS = [8] # This subject has missing channels

    def __init__(self, root: str, dataset_name: Optional[str] = None, dev: bool = False, refresh_cache: bool = False, download=False):
        super().__init__(root, dataset_name, dev, refresh_cache)
        if download:
            _download_ISRUC_group1(root, exclude_subjects=self._EXCLUDE_SUBJECTS)
        
    def process_EEG_data(self):
        raw_dir = os.path.join(self.root, 'raw')
        subject_ids = os.listdir(raw_dir)
        if self.dev:
            subject_ids = subject_ids[:5]
        subjects = {
            subject_id: [
                {
                    "load_from_path": raw_dir,
                    "signal_file": f'{subject_id}/{subject_id}.edf',
                    "label1_file": f'{subject_id}/{subject_id}_1.txt',
                    "label2_file": f'{subject_id}/{subject_id}_2.txt',
                    "save_to_path": self.filepath,
                    'subject_id': subject_id,
                }
            ]
            for subject_id in subject_ids
        }
        return subjects

if __name__ == '__main__':
    dataset = ISRUCDataset(
        root='/srv/scratch1/data/ISRUC-I',
        dev=True,
        refresh_cache=True,
        download=True,
    )
        
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
    
    from pyhealth.tasks import sleep_staging_isruc_fn
    sleep_staging_ds = dataset.set_task(sleep_staging_isruc_fn)