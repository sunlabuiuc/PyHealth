import os
import string

def biview_multisent_fn(patient):
    """ Processes single patient for xray report generation"""
    sample = {}
    image_path = []
    img_root = "/srv/local/data/IU_XRay/images/images_normalized"

    for data in patient:
        patient_id = data["patient_id"]
        if data["view"] == "frontal":
            image_path.insert(0,os.path.join(img_root, data["path"]))
        if data["view"] == "lateral":
            image_path.append(os.path.join(img_root, data["path"]))

        impression = data["impression"]
        findings = data["findings"]
        report = f"{impression} . {findings}"
    
    sample["patient_id"] = patient_id
    sample["image_path"] = image_path

    sents = report.lower().split(".")
    sents = [sent for sent in sents if len(sent.strip()) > 1]
    sample["caption"] = []
    for isent, sent in enumerate(sents):
        tokens = sent.translate(str.maketrans("", "", string.punctuation)) \
                     .strip() \
                     .split()
        sample["caption"].append([".", *[token for token in tokens],"."])

    if sample["caption"] == []:
        sample["caption"] = [[" "," "]]

    sample["caption"][0][0] = "<start>"
    sample["caption"][-1].append("<end>")

    return [sample]