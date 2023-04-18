import os

def biview_onesent_fn(patient):
    """ Processes single patient for xray report generation"""
    sample = {}
    sample['frontal_image_path'] = None
    sample['lateral_image_path'] = None
    img_root = '/srv/local/data/IU_XRay/images/images_normalized'

    for data in patient:
        sample['patient_id'] = data['patient_id']
        sample['report'] = [data['impression']+data['findings']]
        if data['view'] == 'frontal':
            sample['frontal_image_path'] = os.path.join(img_root, data['path'])
        if data['view'] == 'lateral':
            sample['lateral_image_path'] = os.path.join(img_root, data['path'])
        patient = [sample]
        return patient