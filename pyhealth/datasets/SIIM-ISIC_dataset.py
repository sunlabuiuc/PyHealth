# Author: Neel Pai
# Netid: ipai3
# Pulled from Original Paper: Representing visual classification as a linear combination of words
# Paper link: https://arxiv.org/abs/2311.10933
# Note: Dataset is publicly available outside of usage in this paper
# Description: PyHealth dataset class for SIIM-ISIC DICOM data file classification. Contains melanoma
#              imaging data tied to a variety of characteristics, such as sex, age_approx, 
#              part of body, diagnosis, and a benign/malignant flag,

from .base_dataset import BaseDataset
import polars as pl

class SIIM_ISIC_Dataset(BaseDataset):


    """Base image dataset for SIIM-ISIC melanoma data

    Dataset is available at:
    https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data

    
    Args:
        root: Root directory of the raw data containing the dataset files.
    """
     
    def __init__(self, root: str, dev: bool = False):
        super().__init__(
            root=root,
            tables=["metadata"],
            dataset_name="SIIM_ISIC_Dataset",
            config_path="configs/melanoma.yaml",
            dev=dev,
        )

    def preprocess_metadata(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Add full DICOM path to each row
        image_path_expr = pl.concat_str(
            [pl.lit(f"{self.root}/dicoms/"), pl.col("image_name"), pl.lit(".dcm")]
        ).alias("image_path")
        return df.with_columns(image_path_expr)
    
def main():
    dataset = SIIM_ISIC_Dataset(root="/path/to/melanoma", dev=True)
    dataset.stats()

    patient = next(dataset.iter_patients())
    print(patient.patient_id)
    print(patient.data_source.head())  # Will show metadata + image_path

if __name__ == "__main__":
    main()


