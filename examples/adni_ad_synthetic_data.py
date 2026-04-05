"""Synthetic ADNI data helper.

Author: Bryan Lau (bryan16@illinois.edu)
Description:
    This helper function creates an ADNI image record along with the required 
    directory structure and metadata files.
"""
import nibabel as nib
import numpy as np
import random

def create_adni_image(adni_root_path, subject_id=None, group=None):
    """Create test ADNI directory structure populated with synthetic data.

    Creates a directory structure for one subject, with the same layout as 
    the one obtained by downloading actual ADNI data files.

    The directory structure has the following layout:

    - root
        - subject id
            - pre-processing transform
                - date acquired
                    - image uid
                        MRI image file
        metadata xml file

    Args:
        adni_root_path: Path in which to create the ADNI directory structure
        subject_id: Subject ID for this directory structure, if None then a 
            random Subject ID will be generated instead.
        group: Label to assign to this subject, if None then a label will be 
            randomly selected from the three valid choices (i.e. MCI, CN, AD).

    Returns:
        Dictionary containing the following values for later comparison:
        - subject_id: Subject ID of the patient.
        - group: Label assigned to the patient.
        - gender: Patient's randomly selected gender.
        - age: Patient's randomly selected age.
        - weight: Patient's randomly selected weight.
        - image_uid: Unique ID of the MRI image.
        - image_path: Path to the MRI image file.
    """

    if not subject_id:
        subject_id = f"002_S_{random.randint(0, 9999):04d}"
    if not group:
        group = random.choice(["CN", "MCI", "AD"])
    gender = random.choice(["M", "F"])
    age = round(random.uniform(40.0000, 85.0000), 4)
    weight = round(random.uniform(55.0, 120.0), 1)
    date_acquired = f"{random.randint(1950, 2000)}-03-15"

    xform_str = "MPR__GradWarp__B1_Correction__N3"
    date_dir = f"{date_acquired}_09_45_30.0"
    series_id = f"{random.randint(0, 99999):05d}"
    image_uid = f"{random.randint(0, 99999):05d}"

    # Create MRI image directory structure
    adni_image_dir = adni_root_path / subject_id / \
        xform_str / date_dir / f"I{image_uid}"
    adni_image_dir.mkdir(parents=True)

    # Generate test image filename and data
    file_date_str = f"{date_acquired.replace("-", "")}{random.randint(100000000, 300000000):9d}"
    image_filepath = adni_image_dir / \
        f"ADNI_{subject_id}_MR_{xform_str}_Br_{file_date_str}_S{series_id}_I{image_uid}.nii"
    image_data = np.random.rand(121, 145, 121).astype(np.float32)

    # Generate group marking (to simulate image features)
    mark_value = 10.0
    if group == "CN":
        image_data[10:15, 10:15, 10:15] = mark_value
    elif group == "MCI":
        image_data[60:65, 60:65, 60:65] = mark_value
    elif group == "AD":
        image_data[100:105, 100:105, 100:105] = mark_value

    # Save the image
    mri_image = nib.Nifti1Image(image_data, affine=np.eye(4))
    nib.save(mri_image, image_filepath)

    # Generate metadata xml
    metadata_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <idaxs>
        <project>
            <projectIdentifier>ADNI</projectIdentifier>
            <siteKey>002</siteKey>
            <subject>
                <subjectIdentifier>{subject_id}</subjectIdentifier>
                <researchGroup>{group}</researchGroup>
                <subjectSex>{gender}</subjectSex>
                <study>
                    <subjectAge>{age}</subjectAge>
                    <weightKg>{weight}</weightKg>
                    <series>
                        <seriesIdentifier>{series_id}</seriesIdentifier>
                        <dateAcquired>{date_acquired}</dateAcquired>
                        <seriesLevelMeta>
                            <derivedProduct>
                                <imageUID>{image_uid}</imageUID>
                            </derivedProduct>
                        </seriesLevelMeta>
                    </series>
                </study>
            </subject>
        </project>
    </idaxs>
    """
    metadata_xml_filename = f"ADNI_{subject_id}_{xform_str}_S{series_id}_I{image_uid}.xml"
    metadata_xml_path = adni_root_path / metadata_xml_filename
    with open(metadata_xml_path, "w", encoding="utf-8") as f:
        f.write(metadata_xml)

    # Return test values for later comparison
    return {
        "subject_id": subject_id,
        "group": group,
        "gender": gender,
        "age": age,
        "weight": weight,
        "image_uid": image_uid,
        "image_path": image_filepath,
    }