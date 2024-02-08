import os
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pyhealth.data.data import Event
from pyhealth.datasets.utils import hash_str
from pyhealth.synthetic.halo.halo import HALO
from pyhealth.synthetic.halo.generator import Generator
from pyhealth.synthetic.halo.processor import Processor

basedir = '/shared/bpt3/data/FairPlay/MIMIC'

##########
# CONFIG #
##########

# mortality
# ethnicity
# insurance
# gender
# ethnicityAndInsurance
key = 'ethnicityAndInsurance'

# 0
# 1
# 2
# 3
# 4
fold = 0

# mortality: 0-1
# ethniciy: 0-9
# insurance: 0-5
# gender: 0-3
# ethnicityAndInsurance: 0-29
label_idx = [24, 25, 26]

device = 'cuda:0'

############
# CONTENTS #
############

ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.0/hosp/"
dataset_name = "MIMIC4-demo"
tables = ["diagnoses_icd", "labevents"]
code_mapping = {"NDC": "RxNorm"}
dev = False
args_to_hash = (
    [dataset_name, ROOT]
    + sorted(tables)
    + sorted(code_mapping.items())
    + ["dev" if dev else "prod"]
)
# d9288d770061592c27878a53d7c2c263.pkl
filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
cache_path = f'{basedir}/cache'
dataset_filepath = os.path.join(cache_path, filename)
dataset = None

# Event Handlers
def _normalize_key(key):
    """
    In floating-point arithmetic, nan is a special value which, according to the IEEE standard, does not compare equal to any other float value, including itself. 
    This means that if you have multiple nan values as keys in a dictionary, each will be treated as a unique key, because nan != nan.
    """
    if isinstance(key, float) and math.isnan(key):
        return 'nan'  # Convert NaN to a string for consistent key comparison
    return key

def diagnoses_icd_handler(event: Event):
    if "ICD9" in event.vocabulary:
        split_code = event.code.split('.')
        assert len(split_code) <= 2
        return f"{split_code[0]}_{event.vocabulary}"
    else:
        None
        
def reverse_diagnoses_icd(event: str):
    return {
        "table": "diagnoses_icd",
        "code": event[0].split("_")[0],
        "vocabulary": event[0].split("_")[1],
    }

def procedures_icd_handler(event: Event):
    # some NDC --> RxNorm do not exist; those codes will be NDC
    if "ICD9" in event.vocabulary:
        split_code = event.code.split('.')
        assert len(split_code) <= 2
        return f"{split_code[0]}_{event.vocabulary}"
    else:
        None
        
def reverse_procedures_icd(event: str):
    return {
        "table": "procedures_icd",
        "code": event[0].split("_")[0],
        "vocabulary": event[0].split("_")[1],
    }

def prescriptions_handler(event: Event):
    # some NDC --> RxNorm do not exist; those codes will be NDC
    if "RxNorm" in event.vocabulary:
        return f"{event.code}_{event.vocabulary}"
    else:
        None
        
def reverse_prescriptions(event: str):
    return {
        "table": "prescriptions",
        "code": event[0].split("_")[0],
        "vocabulary": event[0].split("_")[1],
    }

def make_lab_global_event(event: Event):
    lab_name = event.code
    lab_value = event.attr_dict['value']
    lab_unit = event.attr_dict['unit']
    return (lab_name, lab_value, lab_unit)

def make_lab_numeric(event: Event):
    lab_value = event.attr_dict['value']
    if (type(lab_value) == str):
        try:
            lab_value = float(lab_value)
        except Exception as e:
            lab_value = np.nan

    if (np.isnan(lab_value)):
        lab_value = None

    # data flitering/cleaning for MIMIC4
    if (event.attr_dict['unit'] == ' '):
        lab_value = None

    return (lab_value)

def make_lab_event_id(event: Event):
    lab_name = event.code
    lab_unit = event.attr_dict['unit']
    return (lab_name, _normalize_key(lab_unit))

def lab_event_id(event: Event, bin_index: int):
    id_info = make_lab_event_id(event)
    lab_value = bin_index
    return (*id_info, lab_value)

def reverse_labevents(event: str, processor: Processor):
    bins = processor.event_bins['lab'][(event[0], event[1])]
    return {
        "table": "labevents",
        "code": event[0],
        "vocabulary": "MIMIC4_LABNAME",
        "attr_dict": {
            "value": np.random.uniform(bins[event[2]], bins[event[2]+1]),
            "unit": event[1],
        }
    }

event_handlers = {}
event_handlers["diagnoses_icd"] = diagnoses_icd_handler # just use the .code field
event_handlers["procedures_icd"] = procedures_icd_handler # just use the .code field
event_handlers["prescriptions"] = prescriptions_handler # uses NDC code by default
event_handlers["labevents"] = make_lab_numeric # default handler applied

discrete_event_handlers = {}
discrete_event_handlers["labevents"] = lab_event_id

reverse_event_handlers = {}
reverse_event_handlers["diagnoses_icd"] = reverse_diagnoses_icd
reverse_event_handlers["procedures_icd"] = reverse_procedures_icd
reverse_event_handlers["prescriptions"] = reverse_prescriptions
reverse_event_handlers["labevents"] = reverse_labevents

# histogram for lab values
compute_histograms = ["labevents"]
size_per_event_bin = {"labevents": 10}
hist_identifier = {'labevents': make_lab_event_id }

# Label Functions
mortality_label_fn_output_size = 1
def mortality_label_fn(**kwargs):
    pdata = kwargs['patient_data']
    return (1,) if pdata.death_datetime else (0,) # 1 for dead, 0 for alive

def reverse_mortality_label_fn(label_vec):
    return {
        'death_datetime': datetime.datetime.now() if label_vec == 1 else None
    }

gender_label_fn_output_size = 3
def gender_label_fn(**kwargs):
    pdata = kwargs['patient_data']
    mortality_idx = [1] if pdata.death_datetime else [0]
    gender_idx = [1, 0] if pdata.gender == 'M' else [0, 1]
    return tuple(mortality_idx + gender_idx)
    
def reverse_gender_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    gender_idx = label_vec[1:3]
    return {
        'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
        'gender': 'M' if gender_idx[0] == 1 else 'F'
    } 

ethnicity_map = {
    'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC/LATINO',
    'WHITE': 'WHITE',
    'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
    'BLACK/AFRICAN AMERICAN': 'BLACK',
    'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC/LATINO',
    'UNABLE TO OBTAIN': 'OTHER/UNKNOWN',
    'ASIAN - KOREAN': 'ASIAN',
    'BLACK/CARIBBEAN ISLAND': 'BLACK',
    'BLACK/AFRICAN': 'BLACK',
    'WHITE - EASTERN EUROPEAN': 'WHITE',
    'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER/UNKNOWN', # TODO: Check This One
    'MULTIPLE RACE/ETHNICITY': 'OTHER/UNKNOWN',
    'WHITE - OTHER EUROPEAN': 'WHITE',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER/UNKNOWN', # TODO: Check This One
    'PORTUGUESE': 'WHITE',
    'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - MEXICAN': 'HISPANIC/LATINO',
    'UNKNOWN': 'OTHER/UNKNOWN',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - HONDURAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - CUBAN': 'HISPANIC/LATINO',
    'PATIENT DECLINED TO ANSWER': 'OTHER/UNKNOWN',
    'OTHER': 'OTHER/UNKNOWN',
    'HISPANIC/LATINO - DOMINICAN': 'HISPANIC/LATINO',
    'BLACK/CAPE VERDEAN': 'BLACK',
    'ASIAN': 'ASIAN',
    'ASIAN - ASIAN INDIAN': 'ASIAN',
    'HISPANIC OR LATINO': 'HISPANIC/LATINO',
    'ASIAN - CHINESE': 'ASIAN',
    'WHITE - BRAZILIAN': 'WHITE',
    'SOUTH AMERICAN': 'HISPANIC/LATINO',
    'WHITE - RUSSIAN': 'WHITE',
    'HISPANIC/LATINO - SALVADORAN': 'HISPANIC/LATINO',
}

ethnicity_label_fn_output_size = 6
def ethnicity_label_fn(**kwargs):
    pdata = kwargs['patient_data']
    ethnicity = ethnicity_map[pdata.ethnicity]
    mortality_idx = [1] if pdata.death_datetime else [0]
    ethnicity_idx = [1, 0, 0, 0, 0] if ethnicity == 'WHITE' else [0, 1, 0, 0, 0] if ethnicity == 'BLACK' else [0, 0, 1, 0, 0] if ethnicity == 'HISPANIC/LATINO' else [0, 0, 0, 1, 0] if ethnicity == 'ASIAN' else [0, 0, 0, 0, 1]
    return tuple(mortality_idx + ethnicity_idx)
    
def reverse_ethnicity_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    ethnicity_idx = label_vec[1:6]
    return {
        'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
        'ethnicity': 'WHITE' if ethnicity_idx[0] == 1 else 'BLACK' if ethnicity_idx[1] == 1 else 'HISPANIC/LATINO' if ethnicity_idx[2] == 1 else 'ASIAN' if ethnicity_idx[3] == 1 else 'OTHER/UNKNOWN',
    }
    
insurance_label_fn_output_size = 4
def insurance_label_fn(**kwargs):
    pdata = kwargs['patient_data']
    mortality_idx = [1] if pdata.death_datetime else [0]
    insurance_idx = [1, 0, 0] if pdata.insurance == 'Medicare' else [0, 1, 0] if pdata.insurance == 'Medicaid' else [0, 0, 1]
    return tuple(mortality_idx + insurance_idx)
    
def reverse_insurance_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    insurance_idx = label_vec[1:4]
    return {
        'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
        'insurance': 'Medicare' if insurance_idx[0] == 1 else 'Medicaid' if insurance_idx[1] == 1 else 'Other',
    }
    
ethnicityAndInsurance_label_fn_output_size = 9
def ethnicityAndInsurance_label_fn(**kwargs):
    pdata = kwargs['patient_data']
    ethnicity = ethnicity_map[pdata.ethnicity]
    mortality_idx = [1] if pdata.death_datetime else [0]
    ethnicity_idx = [1, 0, 0, 0, 0] if ethnicity == 'WHITE' else [0, 1, 0, 0, 0] if ethnicity == 'BLACK' else [0, 0, 1, 0, 0] if ethnicity == 'HISPANIC/LATINO' else [0, 0, 0, 1, 0] if ethnicity == 'ASIAN' else [0, 0, 0, 0, 1]
    insurance_idx = [1, 0, 0] if pdata.insurance == 'Medicare' else [0, 1, 0] if pdata.insurance == 'Medicaid' else [0, 0, 1]
    return tuple(mortality_idx + ethnicity_idx + insurance_idx)

def reverse_ethnicityAndInsurance_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    ethnicity_idx = label_vec[1:6]
    insurance_idx = label_vec[6:9]
    return {
        'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
        'ethnicity': 'WHITE' if ethnicity_idx[0] == 1 else 'BLACK' if ethnicity_idx[1] == 1 else 'HISPANIC/LATINO' if ethnicity_idx[2] == 1 else 'ASIAN' if ethnicity_idx[3] == 1 else 'OTHER/UNKNOWN',
        'insurance': 'Medicare' if insurance_idx[0] == 1 else 'Medicaid' if insurance_idx[1] == 1 else 'Other',
    }

label_fn = eval(f'{key}_label_fn')
label_fn_output_size = eval(f'{key}_label_fn_output_size')

processor = Processor(
        dataset=None,
        use_tables=None,
        event_handlers=event_handlers,
        compute_histograms=compute_histograms,
        hist_identifier=hist_identifier,
        size_per_event_bin=size_per_event_bin,
        discrete_event_handlers=discrete_event_handlers,
        size_per_time_bin=10,
        label_fn=label_fn,
        label_vector_len=label_fn_output_size,
        name="HALO-FairPlay-mimic",
        refresh_cache=False,
        expedited_load=True,
        dataset_filepath=dataset_filepath,
        cache_path=cache_path,
        max_visits=20,
    )
model = HALO(
    n_ctx=processor.total_visit_size,
    total_vocab_size=processor.total_vocab_size,
    device=device
)
model.load_state_dict(torch.load(open(f'{basedir}/model_saves/mimic4_halo_{key}_model_{fold}.pt', 'rb'), map_location='cpu')['model'])
model.to(device)

labels = pickle.load(open(f'{basedir}/mimic4_{key}_labels_{fold}.pkl', 'rb'))
for l_idx in tqdm(label_idx):
    if os.path.exists(f'{basedir}/temp/synthetic_{key}_data_{fold}_{l_idx}.pkl') or os.path.exists(f'{basedir}/synthetic_{key}_data_{fold}.pkl'):
        continue
      
    generator = Generator(
            model=model,
            processor=processor,
            batch_size=1,
            device=device,
            save_dir=f'{basedir}/temp',
            save_name=f'synthetic_{key}_data_{fold}_{l_idx}'
        )

    synthetic_dataset = generator.generate_conditioned([labels[l_idx]])
    
if all([os.path.exists(f'{basedir}/temp/synthetic_{key}_data_{fold}_{l_idx}.pkl') for l_idx in range(len(labels))]):
    synthetic_data = [p for l_idx in range(len(labels)) for p in pickle.load(open(f'{basedir}/temp/synthetic_{key}_data_{fold}_{l_idx}.pkl', 'rb'))]
    pickle.dump(synthetic_data, open(f'{basedir}/synthetic_{key}_data_{fold}.pkl', 'wb'))