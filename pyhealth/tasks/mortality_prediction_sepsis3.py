def mortality_prediction_sepsis3_fn(patient):
    """
    Processes a single patient's data for the mortality prediction task with the Sepsis-3 dataset.
    
    This function prepares the data for a binary classification task where the objective is
    to predict whether a patient will decease within 30 days after admission based on 
    clinical information available at the time of their ICU stay.

    Args:
        patient: a dictionary containing a patient's data with keys as column names and values
                 as lists of the patient's data across different visits or time points.

    Returns:
        samples: a list of samples, each sample is a dict with independent labels and 
                 a 'label' key for the target variable.

    The function utilizes the following independent labels from the Sepsis-3 dataset:
    'urineoutput', 'lactate_min', 'bun_mean', 'sysbp_min', 'metastatic_cancer',
    'inr_max', 'age', 'sodium_max', 'aniongap_max', 'creatinine_min', 'spo2_mean'.
    The target label used is 'thirtyday_expire_flag'.

    Example:
        >>> patient_data = {
        ...     'urineoutput': [500],
        ...     'lactate_min': [2.0],
        ...     'bun_mean': [14],
        ...     'sysbp_min': [120],
        ...     'metastatic_cancer': [0],
        ...     'inr_max': [1.1],
        ...     'age': [65],
        ...     'sodium_max': [140],
        ...     'aniongap_max': [12],
        ...     'creatinine_min': [1.0],
        ...     'spo2_mean': [98],
        ...     'thirtyday_expire_flag': [0]
        ... }
        >>> sample = mortality_prediction_sepsis3_fn(patient_data)
        >>> print(sample)
        [{'urineoutput': 500, 'lactate_min': 2.0, 'bun_mean': 14, 'sysbp_min': 120, 
          'metastatic_cancer': 0, 'inr_max': 1.1, 'age': 65, 'sodium_max': 140, 
          'aniongap_max': 12, 'creatinine_min': 1.0, 'spo2_mean': 98, 'label': 0}]
    """
    samples = []
    # Here, assuming each patient's data is already aggregated in a single record
    if patient:
        # Extract the target label
        label = patient.get('thirtyday_expire_flag', [0])[0]
        
        # Prepare the sample dictionary
        sample = {
            'urineoutput': patient.get('urineoutput', [None])[0],
            'lactate_min': patient.get('lactate_min', [None])[0],
            'bun_mean': patient.get('bun_mean', [None])[0],
            'sysbp_min': patient.get('sysbp_min', [None])[0],
            'metastatic_cancer': patient.get('metastatic_cancer', [None])[0],
            'inr_max': patient.get('inr_max', [None])[0],
            'age': patient.get('age', [None])[0],
            'sodium_max': patient.get('sodium_max', [None])[0],
            'aniongap_max': patient.get('aniongap_max', [None])[0],
            'creatinine_min': patient.get('creatinine_min', [None])[0],
            'spo2_mean': patient.get('spo2_mean', [None])[0],
            'label': label
        }
        samples.append(sample)

    return samples