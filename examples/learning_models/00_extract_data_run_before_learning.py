# -*- coding: utf-8 -*-
"""A data script to unzip preprocess CMS and MIMIC demo datasets. Should
run before the learning models.
"""
# License: BSD 2 clause


# environment setting
import os
from zipfile import ZipFile

if __name__ == "__main__":
    # override here to specify where the data locates
    root_dir = ''
    root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(root_dir, 'datasets')
    os.chdir(data_dir)
    print(root_dir, data_dir)

    with ZipFile(os.path.join(data_dir, 'cms.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting cms files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'mimic.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting mimic demo files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'image.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting image files now...')
        zip.extractall()
        print('Done!')