from collections import Counter
import math
import os
import pickle
import time
import numpy as np

import torch
from pyhealth import BASE_CACHE_PATH
from pyhealth.data.data import Event, Patient
from pyhealth.datasets.mimic4 import MIMIC4Dataset
from pyhealth.datasets.utils import hash_str
from pyhealth.medcode.inner_map import InnerMap
from pyhealth.synthetic.halo.evaluator import Evaluator
from pyhealth.synthetic.halo.generator import Generator
from pyhealth.synthetic.halo.halo import HALO
from pyhealth.synthetic.halo.processor import Processor
from pyhealth import logger

from pyhealth.synthetic.halo.trainer import Trainer


dataset_refresh_cache = False # re-compute the pyhealth dataset
processor_redo_processing = True # use cached dataset vocabulary
processor_expedited_reload = False # idk what this does
processor_refresh_qualified_histogram = False # recompute top K histograms for continuous valued events
trainer_from_dataset_save = False # used for caching dataset split (good for big datasets that take a long time to split)
trainer_save_dataset_split = True # used for test reproducibility


if __name__ == "__main__":
    # for debugging:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # wget -r -N -c -np --user bdanek --ask-password https://physionet.org/files/mimiciv/2.2/
    # for file in *.gz; do
        # gunzip "$file"
        # done
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    ROOT = "/home/bdanek2/physionet.org/files/mimiciv/2.2/hosp"
    dataset_name = "MIMIC4-demo"
    tables = ["procedures_icd", "labevents"]
    code_mapping = {"NDC": "RxNorm"}
    dev = False

    # use drug name instead of ndc code
    # need to reduce the space for procedures_icd, prescriptions, ect
    # should use more broad terms for the code mapping
    
    args_to_hash = (
        [dataset_name, ROOT]
        + sorted(tables)
        + sorted(code_mapping.items())
        + ["dev" if dev else "prod"]
    )
    
    filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
    MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
    dataset_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    if not os.path.exists(dataset_filepath):
        dataset = MIMIC4Dataset(
        root=ROOT,
        tables=tables,
        code_mapping=code_mapping,
        refresh_cache=dataset_refresh_cache,
        dev=dev
    )
    else:
        dataset = None

    
    dataset.stat()
    dataset.info()
    event_handlers = {}

    def diagnoses_icd_handler(event: Event):
        if "ICD9" in event.vocabulary:
            return f"{event.code}_{event.vocabulary}"
        else:
            None
    
    def procedures_icd_handler(event: Event):
        # some NDC --> RxNorm do not exist; those codes will be NDC
        if "ICD9" in event.vocabulary:
            return f"{event.code}_{event.vocabulary}"
        else:
            None
    

    def prescriptions_handler(event: Event):
        # some NDC --> RxNorm do not exist; those codes will be NDC
        if "RxNorm" in event.vocabulary:
            return f"{event.code}_{event.vocabulary}"
        else:
            None

    event_handlers["diagnoses_icd"] = diagnoses_icd_handler # just use the .code field
    event_handlers["procedures_icd"] = procedures_icd_handler # just use the .code field
    event_handlers["prescriptions"] = prescriptions_handler # uses NDC code by default

    # histogram for lab values
    compute_histograms = ["labevents"]
    size_per_event_bin = {"labevents": 10}

    def make_lab_global_event(event: Event):
        lab_name = event.code
        lab_value = event.attr_dict['value']
        lab_unit = event.attr_dict['unit']
        return (lab_name, lab_value, lab_unit)
    
    # 1. Collect primitives for the histogram (make numeric)
    def make_lab_numeric(event: Event):
        # for continuous events, the event handler should just turn this into a float.
        # we will process the list of floats into a histogram, and make ids based on the 
        lab_value = event.attr_dict['value']
        if (type(lab_value) == str):
            try:
                lab_value = float(lab_value)
            except Exception as e:
                logger.debug(f"Could not convert lab value to float: {lab_value}, type(lab_value)={type(lab_value)})")
                lab_value = np.nan

        if (np.isnan(lab_value)):
            lab_value = None

        # data flitering/cleaning for MIMIC4
        if (event.attr_dict['unit'] == ' '):
            lab_value = None

        return (lab_value)
    
    event_handlers["labevents"] = make_lab_numeric # default handler applied
    
    # 2. (optional) create an id for lab event which does not contain the value
    #     This is the identfiier for the distribution which a particular type of lab belongs to
    
    def _normalize_key(key):
        """
        In floating-point arithmetic, nan is a special value which, according to the IEEE standard, does not compare equal to any other float value, including itself. 
        This means that if you have multiple nan values as keys in a dictionary, each will be treated as a unique key, because nan != nan.
        """
        if isinstance(key, float) and math.isnan(key):
            return 'nan'  # Convert NaN to a string for consistent key comparison
        return key
    
    def make_lab_event_id(event: Event):
        lab_name = event.code
        lab_unit = event.attr_dict['unit']
        return (lab_name, _normalize_key(lab_unit))
    
    # 3. Create a vocabulary element
    def lab_event_id(event: Event, bin_index: int):
        lab_name = event.code
        lab_value = bin_index
        lab_unit = event.attr_dict['unit']
        return (lab_name, lab_unit, lab_value)

    def naieve_label_fn(patient_data: Patient):
        return 0 if patient_data.death_datetime is None else 1
    
    def reverse_label_fn(label: int):
        return "alive" if label == 0 else "dead"

    label_fn_output_size = 1
    
    processor = Processor(
        dataset=dataset,
        use_tables=None,
        event_handlers=event_handlers,
        compute_histograms=['labevents'],
        hist_identifier={'labevents': make_lab_event_id },
        size_per_event_bin={'labevents': 10},
        discrete_event_handlers={"labevents": lab_event_id},
        size_per_time_bin=10,
        label_fn=naieve_label_fn,
        label_vector_len=label_fn_output_size,
        name="HALO-FairPlay-mimic",
        refresh_cache=processor_redo_processing,
        expedited_load=processor_expedited_reload,
        dataset_filepath=None if dataset is not None else dataset_filepath,
        max_visits=40, # optional parameter cut off the tail of the distribution of visits
        max_continuous_per_table=10
    )

    model = HALO(
        n_ctx=processor.total_visit_size,
        total_vocab_size=processor.total_vocab_size,
        device=device
    )
    print(model.__call__)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(optimizer.__class__)

    basedir = '/home/bdanek2/halo_development/testing_3'
    model_save_name = 'halo_mortality_model'
    synthetic_data_save_name = 'synthetic_data'
    fold=0
    num_folds=5
    batch_size=128
    
    trainer = Trainer(
        dataset=processor.dataset,
        model=model,
        processor=processor,
        optimizer=optimizer,
        checkpoint_dir=f'{basedir}/model_saves',
        model_save_name=f'{model_save_name}_{fold}',
        folds=num_folds
    )
    trainer.load_fold_split(fold, from_save=trainer_from_dataset_save, save=trainer_save_dataset_split)
    # pdb.set_trace()
    start_time = time.perf_counter()
    trainer.train(
        batch_size=batch_size,
        epoch=20,
        patience=3,
        eval_period=float('inf')
    )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print("training time:", run_time, run_time / 60, (run_time / 60) / 60)

    # --- generate synthetic dataset using the best model ---
    state_dict = torch.load(open(trainer.get_model_checkpoint_path(), 'rb'), map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)

    generator = Generator(
        model=model,
        processor=processor,
        batch_size=batch_size,
        device=device,
        save_dir=basedir,
        save_name=f'{synthetic_data_save_name}_{fold}'
    )

    # pdb.set_trace()
    # labels = Counter([naieve_label_fn(patient_data=p) for p in trainer.train_dataset])
    # maxLabel = max(labels.values())
    # labels = [(l, maxLabel-labels[l]) for l in labels]
    labels = [(0, 10000), (1, 10000)]
    label_mapping = {(float(l),): reverse_label_fn(l) for l, _ in labels} # the generator expects tuples & floats for labels
    synthetic_dataset = generator.generate_conditioned(labels)
    
    synthetic_dataset = pickle.load(open(generator.save_path, 'rb'))

    def pathfn(plot_type: str, label: tuple):
        prefix = os.path.join(generator.save_dir, 'plots')

        label = label_mapping[label] if label in label_mapping else 'all_labels'
        path_str = f"{prefix}_{plot_type.replace(' ', '_').lower()}_{label}".replace(",", "").replace(".", "")

        return path_str

    # convert the data for standard format for downstream tasks
    evaluator = Evaluator(generator=generator, processor=processor)
    
    stats = evaluator.evaluate(
        source=trainer.test_dataset,
        synthetic=pickle.load(file=open(generator.save_path, 'rb')),
        get_plot_path_fn=pathfn,
        compare_label=list(label_mapping.keys()),
    )
    print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))
    