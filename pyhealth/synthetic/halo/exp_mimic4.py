import os
import time
import math
import torch
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from collections import Counter

from pyhealth import BASE_CACHE_PATH, logger
from pyhealth.data.data import Event, Patient
from pyhealth.datasets.utils import hash_str
from pyhealth.datasets.mimic4 import MIMIC4Dataset
from pyhealth.synthetic.halo.halo import HALO
from pyhealth.synthetic.halo.trainer import Trainer
from pyhealth.synthetic.halo.evaluator import Evaluator
from pyhealth.synthetic.halo.generator import Generator
from pyhealth.synthetic.halo.processor import Processor

dataset_refresh_cache = False # re-compute the pyhealth dataset
processor_redo_processing = False # use cached dataset vocabulary
processor_expedited_reload = False # idk what this does
processor_refresh_qualified_histogram = False # recompute top K histograms for continuous valued events
trainer_from_dataset_save = True # used for caching dataset split (good for big datasets that take a long time to split)
trainer_save_dataset_split = False # used for test reproducibility

experiment_class = "mimic4"

if __name__ == "__main__":
    # wget -r -N -c -np --user bdanek --ask-password https://physionet.org/files/mimiciv/2.2/
    # for file in *.gz; do
        # gunzip "$file"
        # done
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ROOT = "/home/bdanek2/physionet.org/files/mimiciv/2.2/hosp"
    ROOT = "/srv/local/data/MIMIC-IV/hosp"
    dataset_name = "MIMIC4-demo"
    tables = ["diagnoses_icd", "labevents"]
    code_mapping = {"NDC": "RxNorm"}
    dev = True

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
        dataset.stat()
        dataset.info()
    else:
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
    
    # 2. (optional) create an id for lab event which does not contain the value
    #     This is the identfiier for the distribution which a particular type of lab belongs to
    
    def make_lab_event_id(event: Event):
        lab_name = event.code
        lab_unit = event.attr_dict['unit']
        return (lab_name, _normalize_key(lab_unit))
    
    # 3. Create a vocabulary element
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

    basedir = '/home/bdanek2/halo_development/testing_3'
    # basedir = '/srv/local/data/bpt3/FairPlay'

    label_fn = mortality_label_fn
    reverse_label_fn = reverse_mortality_label_fn
    label_fn_output_size = mortality_label_fn_output_size
    model_save_name = 'halo_mortality_model'
    synthetic_data_save_name = 'synthetic_mortality_data'
    experiment_name = 'mortality'
    
    model_save_name = f'{experiment_class}_{model_save_name}'
    synthetic_data_save_name = f'{experiment_class}_{synthetic_data_save_name}'
    experiment_name = f'{experiment_class}_{experiment_name}'
    
    processor = Processor(
        dataset=dataset,
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
        refresh_cache=processor_redo_processing,
        expedited_load=processor_expedited_reload,
        dataset_filepath=None if dataset is not None else dataset_filepath,
        max_visits=40, # optional parameter cut off the tail of the distribution of visits
        # max_continuous_per_table=10
    )

    print(f"Processor results in vocab len {processor.total_vocab_size}, max visit num: {processor.total_visit_size}")

    # model = HALO(
    #     n_ctx=processor.total_visit_size,
    #     total_vocab_size=processor.total_vocab_size,
    #     device=device
    # )
    # print(model.__call__)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # print(optimizer.__class__)
    # state_dict = torch.load(open(f'{basedir}/model_saves/{model_save_name}.pt', 'rb'), map_location=device)
    # model.load_state_dict(state_dict['model'])
    # model.to(device)
    # optimizer.load_state_dict(state_dict['optimizer'])
    # print("loaded previous model from traing; iterations on previous model:", state_dict['iteration'])



    # --- train model ---
    num_folds=5
    batch_size=128
    
    # trainer = Trainer(
    #     dataset=processor.dataset,
    #     model=model,
    #     processor=processor,
    #     optimizer=optimizer,
    #     checkpoint_dir=f'{basedir}/model_saves',
    #     model_save_name=f'{model_save_name}_{fold}',
    #     folds=num_folds
    # )
    # s = trainer.set_basic_splits(from_save=True, save=True)
    # print('split lengths', [len(_s) for _s in s])
    # trainer.set_fold_splits(from_save=True, save=True)
    
    
    
    
    
    #############################
    # Static (Non-Folded) Setup #
    #############################
    
    # start_time = time.perf_counter()
    # trainer.train(
    #     batch_size=batch_size,
    #     epoch=1000,
    #     patience=3,
    #     eval_period=float('inf')
    # )
    # end_time = time.perf_counter()
    # run_time = end_time - start_time
    # print("training time:", run_time, run_time / 60, (run_time / 60) / 60)

    # # --- generate synthetic dataset using the best model ---
    # state_dict = torch.load(open(trainer.get_model_checkpoint_path(), 'rb'), map_location=device)
    # model.load_state_dict(state_dict['model'])
    # model.to(device)

    # generator = Generator(
    #     model=model,
    #     processor=processor,
    #     batch_size=batch_size,
    #     device=device,
    #     save_dir=basedir,
    #     save_name=synthetic_data_save_name'
    # )

    # labels = Counter([label_fn(patient_data=p) for p in trainer.train_dataset])
    # maxLabel = max(labels.values())
    # labels = [(l, maxLabel-labels[l]) for l in labels]
    # label_mapping = {l: reverse_label_fn(l) for l, _ in labels}
    # synthetic_dataset = generator.generate_conditioned(labels)
    # # synthetic_dataset = pickle.load(open(f'{basedir}/{synthetic_data_save_name}.pkl', 'rb'))

    # def pathfn(plot_type: str, label: tuple):
    #     prefix = os.path.join(generator.save_dir, 'plots')

    #     '_'.join(list(labels[label].values())) if label in labels else 'all_labels'
    #     label = label.replace('.', '').replace('/', '').replace(' ', '').lower()
    #     path_str = f"{prefix}_{plot_type}_{label}"

    #     return path_str

    # # conduct evaluation of the synthetic data w.r.t. it's source
    # evaluator = Evaluator(generator=generator, processor=processor)
    # stats = evaluator.evaluate(
    #     source=trainer.train_dataset,
    #     synthetic=pickle.load(file=open(generator.save_path, 'rb')),
    #     get_plot_path_fn=pathfn,
    #     compare_label=list(label_mapping.keys()),
    # )
    # print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))

    # # --- conversion ---
    # print('converting to all data to uniform pyhealth format')
    # synthetic_pyhealth_dataset = generator.convert_ehr_to_pyhealth(synthetic_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
    # train_evaluation_dataset = evaluator.to_evaluation_format(trainer.train_dataset)
    # # pickle.dump(train_evaluation_dataset, open(f'{basedir}/train_data.pkl', 'wb'))
    # train_pyhealth_dataset = generator.convert_ehr_to_pyhealth(train_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
    # eval_evaluation_dataset = evaluator.to_evaluation_format(trainer.eval_dataset)
    # # pickle.dump(eval_evaluation_dataset, open(f'{basedir}/eval_data.pkl', 'wb'))
    # eval_pyhealth_dataset = generator.convert_ehr_to_pyhealth(eval_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
    # test_evaluation_dataset = evaluator.to_evaluation_format(trainer.test_dataset)
    # # pickle.dump(test_evaluation_dataset, open(f'{basedir}/test_data.pkl', 'wb'))
    # test_pyhealth_dataset = generator.convert_ehr_to_pyhealth(test_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
    # # pickle.dump(synthetic_pyhealth_dataset, open(f'{basedir}/synthetic_pyhealth_dataset.pkl', 'wb'))
    # # pickle.dump(train_pyhealth_dataset, open(f'{basedir}/train_pyhealth_dataset.pkl', 'wb'))
    # # pickle.dump(eval_pyhealth_dataset, open(f'{basedir}/eval_pyhealth_dataset.pkl', 'wb'))
    # # pickle.dump(test_pyhealth_dataset, open(f'{basedir}/test_pyhealth_dataset.pkl', 'wb'))
    # print("done")
    
        ################
    # Folded Setup #
    ################
    
    for fold in tqdm(range(num_folds), desc='Training Folds'):
        model = HALO(
            n_ctx=processor.total_visit_size,
            total_vocab_size=processor.total_vocab_size,
            device=device
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # state_dict = torch.load(open(f'{basedir}/model_saves/{model_save_name}_{fold}.pt', 'rb'), map_location=device)
        # model.load_state_dict(state_dict['model'])
        # model.to(device)
        # optimizer.load_state_dict(state_dict['optimizer'])
        # print("loaded previous model from traing; iterations on previous model:", state_dict['iteration'])
        
        # --- train model ---
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
        
        start_time = time.perf_counter()
        # trainer.train(
        #     batch_size=batch_size,
        #     epoch=40,
        #     patience=3,
        #     eval_period=float('inf')
        # )
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

        labels = Counter([label_fn(patient_data=p) for p in trainer.train_dataset])
        maxLabel = max(labels.values())
        labels = [(l, maxLabel-labels[l]) for l in labels]
        
        # synthetic_dataset = generator.generate_conditioned(labels)

        def pathfn(plot_type: str, label: tuple):
            prefix = os.path.join(generator.save_dir, 'plots')

            '_'.join(list(labels[label].values())) if label in labels else 'all_labels'
            label = label.replace('.', '').replace('/', '').replace(' ', '').lower()
            path_str = f"{prefix}_{plot_type}_{label}"

            return path_str
        
        evaluator = Evaluator(generator=generator, processor=processor)

        stats = evaluator.evaluate(
            source=trainer.train_dataset,
            # synthetic=pickle.load(file=open(generator.save_path, 'rb')),
            synthetic=pickle.load(file=open('/home/bdanek2/halo_development/testing_3/mimic4_synthetic_mortality_data_0.pkl', 'rb')),
            get_plot_path_fn=pathfn,
            compare_label=labels,
        )
        print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))

        break

        # convert the data for standard format for downstream tasks
        evaluator = Evaluator(generator=generator, processor=processor)
        # label_mapping = {l: reverse_label_fn(l) for l, _ in labels}
        # synthetic_pyhealth_dataset = generator.convert_ehr_to_pyhealth(synthetic_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
        if not os.path.exists(f'{basedir}/train_{experiment_name}_data_{fold}.pkl'):
            train_evaluation_dataset = evaluator.to_evaluation_format(trainer.train_dataset)
            pickle.dump(train_evaluation_dataset, open(f'{basedir}/train_{experiment_name}_data_{fold}.pkl', 'wb'))
        # else:
        #     train_evaluation_dataset = pickle.load(open(f'{basedir}/train_data_{fold}.pkl', 'rb'))

        if not os.path.exists(f'{basedir}/eval_{experiment_name}_data_{fold}.pkl'):
            eval_evaluation_dataset = evaluator.to_evaluation_format(trainer.eval_dataset)
            pickle.dump(eval_evaluation_dataset, open(f'{basedir}/eval_{experiment_name}_data_{fold}.pkl', 'wb'))
        # else:
        #     eval_evaluation_dataset = pickle.load(open(f'{basedir}/eval_data_{fold}.pkl', 'rb'))
        
        if not os.path.exists(f'{basedir}/test_{experiment_name}_data_{fold}.pkl'):
            test_evaluation_dataset = evaluator.to_evaluation_format(trainer.test_dataset)
            pickle.dump(test_evaluation_dataset, open(f'{basedir}/test_{experiment_name}_data_{fold}.pkl', 'wb'))
        # else:
        #     test_evaluation_dataset = pickle.load(open(f'{basedir}/test_data_{fold}.pkl', 'rb'))
        
        # train_pyhealth_dataset = generator.convert_ehr_to_pyhealth(train_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
        # eval_pyhealth_dataset = generator.convert_ehr_to_pyhealth(eval_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
        # test_pyhealth_dataset = generator.convert_ehr_to_pyhealth(test_evaluation_dataset, reverse_event_handlers, datetime.datetime.now(), label_mapping)
            
        