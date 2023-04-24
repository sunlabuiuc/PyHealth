import os
import string

def biview_multisent_fn(patient):
    """ Processes single patient for X-ray report generation task

    Xray report generation aims a automatically generating the diagnosis report
    from X-ray images taken from two viewpoints namely - frontal and lateral.

    An X-ray report generally consists of following elements
        -   History, describing the reason for the X-ray
        -   Technique, describing the type of X-ray performed - frontal,lateral
        -   Findings, describing the observations made by the radiologist who
                interpreted the X-ray images
        -   Impression, describing the overall interpretation of X-ray results

    Args:
        patient: a list of dictionary of patient X-ray report with below keys
            -   patient_id: type(int)
                    unique identifier for patient
            -   frontal_img_path: type(str)
                    path to frontal X-ray image
            -   lateral_img_path: type(str)
                    path to lateral X-ray image
            -   findings: type(str)
                    text of X-ray report findings
            -   impression: type(str)
                    text of X-ray report impression

    Returns:
        sample: a list of one sample, each sample is a dict with following keys
            -   patient_id: type(int)
                    unique identifier for patient
            -   image_path_list: type(List)
                    list of frontal and lateral image paths
            -   caption: type(List[List])
                    nested list of sentences,where each inner list represents a
                    single sentence in the X-ray report text(formed by
                    concatenating impression and findings).

    Note:   special tokens "<start>" and "<end>", are added to the begining of
            first and end of last sentence. These are mandatory for the task.
    """
    sample = {}
    patient = patient[0]

    report = f"{patient['impression']} . {patient['findings']}"
    caption = []
    sents = report.lower().split(".")
    sents = [sent for sent in sents if len(sent.strip()) > 1]

    for isent, sent in enumerate(sents):
        tokens = sent.translate(str.maketrans("", "", string.punctuation)) \
                     .strip() \
                     .split()
        caption.append([".", *[token for token in tokens],"."])

    if caption == []:
        caption = [["<start>","<end>"]]
    else:
        caption[0][0] = "<start>"
        caption[-1].append("<end>")

    sample["patient_id"] = int(patient["patient_id"])
    sample["image_path_list"] = [ patient["frontal_img_path"],
                                  patient["lateral_img_path"],
                                ]
    sample["caption"] = caption
    return [sample]