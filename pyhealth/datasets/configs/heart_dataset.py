import pandas as pd
from pyhealth.datasets import BaseDataset
from sklearn.preprocessing import LabelEncoder

class HeartDiseaseDataset(BaseDataset):
    """

    This data was pulled form Kaggle and can be found at: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?select=heart_disease_uci.csv

    The dataset is stored in pyhealth/datasets/rawdata/heart_disease_uci.csv for use of testing. 

    The dataset will allow for users to perform predictive analysis on whether or not patients are at risk for heart problems
        Takes in multiple columns such as age, sex, blood pressure, cholesterol

    This function will return [features_dict, target] pairs 

    """
    def __init__(self, root, dataset_name="heart_disease", target_column="target"):
        df = pd.read_csv(root)
        
        for col in ["id", "dataset"]: # get rid of columns that are not necessary. Data cleanup from original source and 
            if col in df.columns:
                df = df.drop(columns=[col])
        
        if "num" in df.columns and target_column not in df.columns: # manipulate columns where needed
            df = df.rename(columns={"num": target_column})
        
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == bool:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        self.df = df
        self.target_column = target_column
        super().__init__(dataset_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = row[self.target_column]
        X = row.drop(labels=[self.target_column]).to_dict()
        return X, y