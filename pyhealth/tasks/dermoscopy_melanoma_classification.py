# Contributor: [Your Name]
# NetID: [Your NetID]

def DermoscopyMelanomaClassification(patient):
    """Task logic for Melanoma Binary Classification.

    This function defines the task extraction logic for the DermoscopyDataset. 
    It parses a `Patient` object and extracts all associated `dermoscopy` events, 
    mapping the image paths and labels into the sample dictionaries required by 
    the PyHealth training engine.

    Args:
        patient (pyhealth.data.Patient): A PyHealth Patient object containing 
            dermoscopy events.

    Returns:
        List[dict]: A list of sample dictionaries. Each dictionary contains:
            - "patient_id" (str): The unique patient/image identifier.
            - "visit_id" (str): The unique visit identifier.
            - "image" (str): The filepath to the source image.
            - "melanoma" (int): The binary label (1 for malignant, 0 for benign).
            
    Note:
        Because Polars loads CSV columns as strings by default, the label is 
        safely cast to `int(float())` to ensure compatibility with PyTorch loss functions.
    """
    samples = []
    events = patient.get_events("dermoscopy")
    
    for event in events:
        samples.append({
            "patient_id": patient.patient_id,
            "visit_id": patient.patient_id,
            "image": (event.attr_dict["image_path"], event.attr_dict.get("mask_path", "")),
            "melanoma": int(float(event.attr_dict["label"]))
        })
    return samples

DermoscopyMelanomaClassification.task_name = "melanoma_classification"
DermoscopyMelanomaClassification.input_schema = {"image": "image"}
# The output schema defines the problem type for BaseModel and metrics routing.
DermoscopyMelanomaClassification.output_schema = {"melanoma": "binary"} 
DermoscopyMelanomaClassification.pre_filter = lambda global_event_df: global_event_df