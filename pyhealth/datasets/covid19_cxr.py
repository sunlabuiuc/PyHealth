import os
from collections import Counter

import pandas as pd

import sys 
sys.path.append('.')

from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.tasks.covid19_cxr_classification import COVID19CXRClassification


class COVID19CXRDataset(BaseDataset):
    """Base image dataset for COVID-19 Radiography Database

    Dataset is available at https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

    **COVID-19 data:
    -----------------------
    COVID data are collected from different publicly accessible dataset, online sources and published papers.
    -2473 CXR images are collected from padchest dataset[1].
    -183 CXR images from a Germany medical school[2].
    -559 CXR image from SIRM, Github, Kaggle & Tweeter[3,4,5,6]
    -400 CXR images from another Github source[7].

    ***Normal images:
    ----------------------------------------
    10192 Normal data are collected from from three different dataset.
    -8851 RSNA [8]
    -1341 Kaggle [9]

    ***Lung opacity images:
    ----------------------------------------
    6012 Lung opacity CXR images are collected from Radiological Society of North America (RSNA) CXR dataset  [8]

    ***Viral Pneumonia images:
    ----------------------------------------
    1345 Viral Pneumonia data are collected from  the Chest X-Ray Images (pneumonia) database [9]

    Please cite the follwoing two articles if you are using this dataset:
    -M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
    -Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

    **Reference:
    [1] https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711
    [2] https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
    [3] https://sirm.org/category/senza-categoria/covid-19/
    [4] https://eurorad.org
    [5] https://github.com/ieee8023/covid-chestxray-dataset
    [6] https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328
    [7] https://github.com/armiro/COVID-CXNet
    [8] https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    [9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        root: root directory of the raw data (should contain many csv files).
        dataset_name: name of the dataset. Default is the name of the class.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Examples:
        >>> dataset = COVID19CXRDataset(
                root="/srv/local/data/zw12/raw_data/covid19-radiography-database/COVID-19_Radiography_Dataset",
            )
        >>> print(dataset[0])
        >>> dataset.stat()
        >>> dataset.info()
    """

    def process(self):
        # process and merge raw xlsx files from the dataset
        covid = pd.DataFrame(
            pd.read_excel(f"{self.root}/COVID.metadata.xlsx")
        )
        covid["FILE NAME"] = covid["FILE NAME"].apply(
            lambda x: f"{self.root}/COVID/images/{x}.png"
        )
        covid["label"] = "COVID"
        lung_opacity = pd.DataFrame(
            pd.read_excel(f"{self.root}/Lung_Opacity.metadata.xlsx")
        )
        lung_opacity["FILE NAME"] = lung_opacity["FILE NAME"].apply(
            lambda x: f"{self.root}/Lung_Opacity/images/{x}.png"
        )
        lung_opacity["label"] = "Lung Opacity"
        normal = pd.DataFrame(
            pd.read_excel(f"{self.root}/Normal.metadata.xlsx")
        )
        normal["FILE NAME"] = normal["FILE NAME"].apply(
            lambda x: x.capitalize()
        )
        normal["FILE NAME"] = normal["FILE NAME"].apply(
            lambda x: f"{self.root}/Normal/images/{x}.png"
        )
        normal["label"] = "Normal"
        viral_pneumonia = pd.DataFrame(
            pd.read_excel(f"{self.root}/Viral Pneumonia.metadata.xlsx")
        )
        viral_pneumonia["FILE NAME"] = viral_pneumonia["FILE NAME"].apply(
            lambda x: f"{self.root}/Viral Pneumonia/images/{x}.png"
        )
        viral_pneumonia["label"] = "Viral Pneumonia"
        df = pd.concat(
            [covid, lung_opacity, normal, viral_pneumonia],
            axis=0,
            ignore_index=True
        )
        df = df.drop(columns=["FORMAT", "SIZE"])
        df.columns = ["path", "url", "label"]
        for path in df.path:
            # assert os.path.isfile(os.path.join(self.root, path))
            assert os.path.isfile(path)
        # create patient dict
        patients = {}
        for index, row in df.iterrows():
            patients[index] = row.to_dict()
        return patients

    def stat(self):
        super().stat()
        print(f"Number of samples: {len(self.patients)}")
        count = Counter([v['label'] for v in self.patients.values()])
        print(f"Number of classes: {len(count)}")
        print(f"Class distribution: {count}")

    @property
    def default_task(self):
        return COVID19CXRClassification()


if __name__ == "__main__":
    dataset = COVID19CXRDataset(
        root="./data/COVID-19_Radiography_Dataset",
    )
    print(list(dataset.patients.items())[0])
    dataset.stat()
    samples = dataset.set_task()
    print(samples[0])
