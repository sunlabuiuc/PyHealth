import os
from collections import Counter
import pandas as pd
import sys

sys.path.append('.')

from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.tasks.drive_classification import DriveClassification


class DriveDataset(BaseDataset):
    """Base image dataset for Digital Retinal Images for Vessel Extraction

    Dataset is available at https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/data

    **Retinal Vessel Segmentation data:
    -----------------------
    Obtained from a diabetic retinopathy screening program in The Netherlands.
    -40 randomly selected photographs.
    -33 do not show any sign of diabetic retinopathy.
    -7 show signs of mild early diabetic retinopathy.

    ***Abnormal images:
    ----------------------------------------
    -25_training: pigment epithelium changes, probably butterfly maculopathy with pigmented scar in fovea, or choroidiopathy, no diabetic retinopathy or other vascular abnormalities.
    -26_training: background diabetic retinopathy, pigmentary epithelial atrophy, atrophy around optic disk
    -32_training: background diabetic retinopathy
    -03_test: background diabetic retinopathy
    -08_test: pigment epithelium changes, pigmented scar in fovea, or choroidiopathy, no diabetic retinopathy or other vascular abnormalities
    -14_test: background diabetic retinopathy
    -17_test: background diabetic retinopathy

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        root: root directory of the raw data (should contain many two folders test and training containing the folders of images and mask).
        dataset_name: name of the dataset. Default is the name of the class.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Examples:
        >>> dataset = DriveDataset(
                root="./data/Drive",
            )
        >>> print(dataset[0])
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process(self):
        image_postfix = ".tif"
        mask_postfix = ".git"
        test_files = [f"{num:02d}_test{image_postfix}" for num in range(1, 21)]
        test_mask_files = [f"{num:02d}_test_mask{mask_postfix}" for num in range(1, 21)]
        training_files = [f"{num:02d}_training{image_postfix}" for num in range(21, 41)]
        training_mask_files = [f"{num:02d}_training{mask_postfix}" for num in range(21, 41)]

        abnormality = {
            "25_training": ["pigment epithelium changes", "butterfly maculopathy with pigmented scar in fovea", "choroidiopathy", "no diabetic retinopathy or other vascular abnormalities"],
            "26_training": ["background diabetic retinopathy", "pigmentary epithelial atrophy"],
            "32_training": ["background diabetic retinopathy"],
            "03_test": ["background diabetic retinopathy"],
            "08_test": ["pigment epithelium changes", "pigmented scar in fovea", "choroidiopathy", "no diabetic retinopathy or other vascular abnormalities"],
            "14_test": ["background diabetic retinopathy"],
            "17_test": ["background diabetic retinopathy"],
        }
        data = {
            "filename": test_files + training_files,
            "mask_filename": test_mask_files + training_mask_files,
        }
        all_data = pd.DataFrame(data)
        all_data["label"] = all_data["filename"].apply(
            lambda filename:
                abnormality[filename[:-len(image_postfix)]] if filename.endswith(image_postfix) and filename[:-len(image_postfix)] in abnormality
                else ["normal"]
        )

        def get_sub_folder(filename):
            if filename.endswith(image_postfix):
                return "images"
            else:
                return "mask"

        def get_folder(filename):
            if "training" in filename:
                return f"training/{get_sub_folder(filename)}"
            else:
                return f"test/{get_sub_folder(filename)}"


        all_data["folder"] = all_data["filename"].apply(get_folder)
        all_data["path"] = all_data["folder"].apply(lambda folder: f"{self.root}/{folder}")
        all_data["path"] = all_data["path"] + "/" + all_data["filename"]

        for path in all_data.path:
            # assert os.path.isfile(os.path.join(self.root, path))
            assert os.path.isfile(path)
        # create patient dict
        patients = {}
        for index, row in all_data.iterrows():
            patients[index] = row.to_dict()
        return patients

    """
    Statistics of DriveDataset:
    Number of samples: 40
    Number of classes: 8
    Class distribution: Counter({
        'normal': 33, 
        'background diabetic retinopathy': 5, 
        'pigment epithelium changes': 2, 
        'choroidiopathy': 2, 
        'no diabetic retinopathy or other vascular abnormalities': 2, 
        'pigmented scar in fovea': 1, 
        'butterfly maculopathy with pigmented scar in fovea': 1, 
        'pigmentary epithelial atrophy': 1
    })
    """
    def stat(self):
        super().stat()
        print(f"Number of samples: {len(self.patients)}")
        count = Counter([label for v in self.patients.values() for label in v['label']])
        print(f"Number of classes: {len(count)}")
        print(f"Class distribution: {count}")

    @property
    def default_task(self):
        return DriveClassification()


if __name__ == "__main__":
    dataset = DriveDataset(
        root="./data/Drive",
    )
    print(list(dataset.patients.items())[0])
    dataset.stat()
    samples = dataset.set_task()
    print(samples[0])
