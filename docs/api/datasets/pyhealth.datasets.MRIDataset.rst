pyhealth.datasets.MRIDataset
===================================

The dataset used is the OASIS MRI dataset (https://sites.wustl.edu/oasisbrains/), which consists of 80,000 brain MRI images. The images have been divided into four classes based on Alzheimer's progression. The dataset aims to provide a valuable resource for analyzing and detecting early signs of Alzheimer's disease.

To make the dataset accessible, the original .img and .hdr files were converted into Nifti format (.nii) using FSL (FMRIB Software Library). The converted MRI images of 461 patients have been uploaded to a GitHub repository, which can be accessed in multiple parts.

For the neural network training, 2D images were used as input. The brain images were sliced along the z-axis into 256 pieces, and slices ranging from 100 to 160 were selected from each patient. This approach resulted in a comprehensive dataset for analysis.

Patient classification was performed based on the provided metadata and Clinical Dementia Rating (CDR) values, resulting in four classes: demented, very mild demented, mild demented, and non-demented. These classes enable the detection and study of different stages of Alzheimer's disease progression.

During the dataset preparation, the .nii MRI scans were converted to .jpg files. Although this conversion presented some challenges, the files were successfully processed using appropriate tools. The resulting dataset size is 1.3 GB.

With this comprehensive dataset, the project aims to explore various neural network models and achieve optimal results in Alzheimer's disease detection and analysis.

Refer to `doc <https://www.kaggle.com/datasets/ninadaithal/imagesoasis>`_ for more information. 

.. autoclass:: pyhealth.datasets.MRIDataset
    :members:
    :undoc-members:
    :show-inheritance: