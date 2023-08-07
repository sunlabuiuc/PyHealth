import os
from collections import Counter
import pandas as pd
from tqdm import tqdm

from base_dataset_v2 import BaseDataset# from pyhealth.datasets.base_dataset_v2 import BaseDataset
from tasks.chexpert_v1_classification import CheXpertV1Classification

class CheXpertV1Dataset(BaseDataset):
    """Base image dataset for CheXpert Database

    Dataset is available at https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2

    **CheXpert v1 data: 
    -----------------------
    - Train: 223414 images from 64540 patients
    - Validation: 902 images from 700 patients
    
    The CheXpert dataset consists of 14 labeled observations (pathology):
    - No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia,
      Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
    For each observation (pathology), there are 4 status:
    - positive (1), negative (0), uncertain (-1), unmentioned (2)

    Please cite the follwoing articles if you are using this dataset:
    - Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R.,
      Shpanskaya, K. and Seekins, J., 2019, July. Chexpert: A large chest radiograph dataset with uncertainty labels
      and expert comparison. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 590-597).

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (The parent directory of /CheXpert-v1.0). *You can choose to use the path to Cassette portion or the Telemetry portion.*
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
    
        >>> dataset = CheXpertV1Dataset(
                root="/home/wuzijian1231/Datasets",
            )
        >>> print(dataset.patients[0])
        >>> dataset.stat()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def process(self):
        # process and merge raw xlsx files from the dataset
        df = pd.DataFrame(
            pd.read_csv(f"{self.root}/CheXpert-v1.0/train.csv")
        )
        df.fillna(value=2.0, inplace=True) # positive (1), negative (0), uncertain (-1), unmentioned (2)
        df["Path"] = df["Path"].apply(
            lambda x: f"{self.root}/{x}"
        )
        df = df.drop(columns=["Sex", "Age", "Frontal/Lateral", "AP/PA"])
        self.pathology = [c for c in df]
        del self.pathology[0]
        df_list= []
        for p in self.pathology:
            df_list.append(df[p])
        self.df_label = pd.concat(df_list, axis=1)
        labels = self.df_label.values.tolist()        
        df.columns = [col for col in df]
        for path in tqdm(df.Path):
            assert os.path.isfile(path)
        # create patient dict
        patients = {}
        for index, row in tqdm(df.iterrows()):
            patients[index] = {'path':row['Path'], 'label':labels[index]}
        return patients

    def stat(self):
        super().stat()
        print(f"Number of samples: {len(self.patients)}")
        print(f"Number of Pathology: {len(self.pathology)}")
        count = {}
        for p in self.pathology:
            cn = self.df_label[p]
            count[p] = Counter(cn)
        for p in self.pathology:
            print(f"Class distribution - {p}: {count[p]}")

    @property
    def default_task(self):
        return CheXpertV1Classification()    

if __name__ == "__main__":
    dataset = CheXpertV1Dataset(
        root="/home/wuzijian1231/Datasets",
    )
    print(dataset.patients[0])
    dataset.stat()
    samples = dataset.set_task()
    print(samples[0])
    