import os
import time
import math
import torch
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from collections import Counter

from pyhealth import BASE_CACHE_PATH
from pyhealth.data import Event 
from pyhealth.datasets.utils import hash_str
from pyhealth.datasets.eicu import eICUDataset
from pyhealth.synthetic.halo.halo import HALO
from pyhealth.synthetic.halo.trainer import Trainer
from pyhealth.synthetic.halo.evaluator import Evaluator
from pyhealth.synthetic.halo.generator import Generator
from pyhealth.synthetic.halo.processor import Processor

dataset_refresh_cache = False # re-compute the pyhealth dataset
dataset_dev = False
processor_redo_processing = False # use cached dataset vocabulary
processor_expedited_reload = False # idk what this does
processor_refresh_qualified_histogram = False # recompute top K histograms for continuous valued events
trainer_from_dataset_save = True # used for caching dataset split (good for big datasets that take a long time to split)
trainer_save_dataset_split = False # used for test reproducibility

experiment_class = 'eicu'

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/"
    ROOT = "/srv/local/data/physionet.org/files/eicu-crd/2.0/"
    # ROOT = "/home/bpt3/data/physionet.org/files/eicu-crd/2.0"
    dataset_name = "eICU-demo"
    tables = ["diagnosis", "lab"]
    code_mapping = {}
    dev = dataset_dev
    
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
        print("dataset doesn't exist - computing")
    else:
        print("loading cached dataset")

    dataset = eICUDataset(
        dataset_name=dataset_name,
        root=ROOT,
        tables=tables,
        code_mapping=code_mapping,
        dev=dev,
        refresh_cache=dataset_refresh_cache,
    )
    dataset.stat()
    dataset.info()

      
    # Event Handlers
    def _normalize_key(key):
        """
        In floating-point arithmetic, nan is a special value which, according to the IEEE standard, does not compare equal to any other float value, including itself. 
        This means that if you have multiple nan values as keys in a dictionary, each will be treated as a unique key, because nan != nan.
        """
        if isinstance(key, float) and math.isnan(key):
            return 'nan'  # Convert NaN to a string for consistent key comparison
        return key
    
    def handle_diagnosis(event: Event):
        """to reduce the complexity of the model, in this example we will convert granular ICD codes to more broad ones (ie 428.3 --> 428)"""
        split_code = event.code.split('.')
        assert len(split_code) <= 2
        return split_code[0]
    
    def reverse_diagnosis(event: str):
        return {
            'table': 'diagnosis',
            'code': event[0],
            'vocabulary': 'ICD9CM',
        }
    
    # these values will be used to compute histograms
    def handle_lab(event: Event):
        """a method for used to convert the lab event into a numerical value; this value will be discretized and serve as the basis for computing a histogram"""
        value = float(event.attr_dict['lab_result'])
        return value
    
    """this callable serves the purpose of generating a unique ID for an event within a particular table (in this case `lab`); 
    It is beneficial to compute histograms on a per-event basis, since the ranges of continuous values for each event type may vary significantly.
    """
    # compute a histogram for each lab name, lab unit pair
    def make_lab_event_id(e: Event):
        return (e.code, _normalize_key(e.attr_dict['lab_measure_name_system']))
    
    # this event handler is called after the histograms have been computed
    """This function serves the purpose of generating a vocabulary element. 
    The vocab element must be hashable, and it is acceptable to use a string to serialize data
    The bin index parameter, is the index within the histogram for this particular lab event.
    """
    def handle_discrete_lab(event: Event, bin_index: int):
        id_info = make_lab_event_id(event)
        lab_value = bin_index
        return (*id_info, lab_value)
    
    def reverse_lab(event: tuple, processor: Processor):
        bins = processor.event_bins['lab'][(event[0], event[1])]
        return {
            'table': 'lab',
            'code': event[0],
            'vocabulary': 'eICU_LABNAME',
            'attr_dict': {
                'lab_result': np.random.uniform(bins[event[2]], bins[event[2]+1]),
                'lab_measure_name_system': event[1],
            }
        }

    # define value handlers; these handlers serve the function of converting an event into a primitive value. 
    # event handlers are called to clean up values
    event_handlers = {}
    event_handlers['diagnosis'] = handle_diagnosis
    event_handlers['lab'] = handle_lab 

    # discrete event handlers are called to produce primitives for auto-discretization
    discrete_event_handlers = {}
    discrete_event_handlers['lab'] = handle_discrete_lab
    
    reverse_event_handlers = {}
    reverse_event_handlers['diagnosis'] = reverse_diagnosis
    reverse_event_handlers['lab'] = reverse_lab
    
    # histogram for lab values
    compute_histograms=['lab']
    size_per_event_bin={'lab': 10}
    hist_identifier={'lab': make_lab_event_id}
        
    
    # Label Functions
    full_label_fn_output_size = 13
    def full_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        mortality_idx = [1] if pdata.death_datetime else [0]
        age = (sorted(pdata.visits.values(), key=lambda v: v.encounter_time)[0].encounter_time - pdata.birth_datetime).days // 365
        age_idx = [1, 0, 0] if age <= 18 else [0, 1, 0] if age < 75 else [0, 0, 1]
        gender_idx = [1, 0, 0] if pdata.gender == 'Male' else [0, 1, 0] if pdata.gender == 'Female' else [0, 0, 1]
        ethnicity_idx = [1, 0, 0, 0, 0, 0] if pdata.ethnicity == 'Caucasian' else [0, 1, 0, 0, 0, 0] if pdata.ethnicity == 'African American' else [0, 0, 1, 0, 0, 0] if pdata.ethnicity == 'Hispanic' else [0, 0, 0, 1, 0, 0] if pdata.ethnicity == 'Asian' else [0, 0, 0, 0, 1, 0] if pdata.ethnicity == 'Native American' else [0, 0, 0, 0, 0, 1]
        return tuple(mortality_idx + age_idx + gender_idx + ethnicity_idx)

    def reverse_full_label_fn(label_vec):
        mortality_idx = label_vec[:1]
        age_idx = label_vec[1:4]
        gender_idx = label_vec[4:7]
        ethnicity_idx = label_vec[7:]
        return {
            'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
            'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly',
            'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown',
            'ethnicity': 'Caucasian' if ethnicity_idx[0] == 1 else 'African American' if ethnicity_idx[1] == 1 else 'Hispanic' if ethnicity_idx[2] == 1 else 'Asian' if ethnicity_idx[3] == 1 else 'Native American' if ethnicity_idx[4] == 1 else 'Other/Unknown',
        }
        
    mortality_label_fn_output_size = 1
    def mortality_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        return (1,) if pdata.death_datetime else (0,) # 1 for dead, 0 for alive

    def reverse_mortality_label_fn(label_vec):
        return {
            'death_datetime': datetime.datetime.now() if label_vec == 1 else None
        }
    
    age_label_fn_output_size = 4
    def age_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        mortality_idx = [1] if pdata.death_datetime else [0]
        age = (sorted(pdata.visits.values(), key=lambda v: v.encounter_time)[0].encounter_time - pdata.birth_datetime).days // 365
        age_idx = [1, 0, 0] if age <= 18 else [0, 1, 0] if age < 75 else [0, 0, 1]
        return tuple(mortality_idx + age_idx)
        
    def reverse_age_label_fn(label_vec):
        mortality_idx = label_vec[:1]
        age_idx = label_vec[1:4]
        return {
            'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
            'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly'
        }
    
    # all labels the `age_label_fn` would generate
    age_label_compare_labels = (
        tuple([1] + [1, 0, 0]),
        tuple([1] + [0, 1, 0]),
        tuple([1] + [0, 0, 1]),
        tuple([0] + [1, 0, 0]),
        tuple([0] + [0, 1, 0]),
        tuple([0] + [0, 0, 1]),
    )
       
    gender_label_fn_output_size = 4 
    def gender_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        mortality_idx = [1] if pdata.death_datetime else [0]
        gender_idx = [1, 0, 0] if pdata.gender == 'Male' else [0, 1, 0] if pdata.gender == 'Female' else [0, 0, 1]
        return tuple(mortality_idx + gender_idx)
        
    def reverse_gender_label_fn(label_vec):
        mortality_idx = label_vec[:1]
        gender_idx = label_vec[1:4]
        return {
            'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
            'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown'
        } 
        
    ethnicity_label_fn_output_size = 7
    def ethnicity_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        mortality_idx = [1] if pdata.death_datetime else [0]
        ethnicity_idx = [1, 0, 0, 0, 0, 0] if pdata.ethnicity == 'Caucasian' else [0, 1, 0, 0, 0, 0] if pdata.ethnicity == 'African American' else [0, 0, 1, 0, 0, 0] if pdata.ethnicity == 'Hispanic' else [0, 0, 0, 1, 0, 0] if pdata.ethnicity == 'Asian' else [0, 0, 0, 0, 1, 0] if pdata.ethnicity == 'Native American' else [0, 0, 0, 0, 0, 1]
        return tuple(mortality_idx + ethnicity_idx)
        
    def reverse_ethnicity_label_fn(label_vec):
        mortality_idx = label_vec[:1]
        ethnicity_idx = label_vec[1:]
        return {
            'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
            'ethnicity': 'Caucasian' if ethnicity_idx[0] == 1 else 'African American' if ethnicity_idx[1] == 1 else 'Hispanic' if ethnicity_idx[2] == 1 else 'Asian' if ethnicity_idx[3] == 1 else 'Native American' if ethnicity_idx[4] == 1 else 'Other/Unknown',
        }
        
    genderAndAge_label_fn_output_size = 7
    def genderAndAge_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        mortality_idx = [1] if pdata.death_datetime else [0]
        age = (sorted(pdata.visits.values(), key=lambda v: v.encounter_time)[0].encounter_time - pdata.birth_datetime).days // 365
        age_idx = [1, 0, 0] if age <= 18 else [0, 1, 0] if age < 75 else [0, 0, 1]
        gender_idx = [1, 0, 0] if pdata.gender == 'Male' else [0, 1, 0] if pdata.gender == 'Female' else [0, 0, 1]
        return tuple(mortality_idx + age_idx + gender_idx)
      
    def reverse_genderAndAge_label_fn(label_vec):
        mortality_idx = label_vec[:1]
        age_idx = label_vec[1:4]
        gender_idx = label_vec[4:7]
        return {
            'death_datetime': datetime.datetime.now() if mortality_idx[0] == 1 else None,
            'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly',
            'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown',
        }
        
        
        
    basedir = '/home/bdanek2/halo_development/testing_eICU'
    # basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'
    # basedir = '/srv/local/data/bpt3/FairPlay'
    
    label_fn = age_label_fn
    reverse_label_fn = reverse_age_label_fn
    label_fn_output_size = age_label_fn_output_size
    model_save_name = 'halo_age_model'
    synthetic_data_save_name = 'synthetic_age_data'
    experiment_name = 'age'
    
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
        name="HALO-FairPlay-eicu",
        refresh_cache=processor_redo_processing,
        expedited_load=processor_expedited_reload,
        dataset_filepath=None if dataset is not None else dataset_filepath,
        max_visits=None,
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
    num_folds = 5
    batch_size = 512
    
    # trainer = Trainer(
    #     dataset=processor.dataset,
    #     model=model,
    #     processor=processor,
    #     optimizer=optimizer,
    #     checkpoint_dir=f'{basedir}/model_saves',
    #     model_save_name=model_save_name,
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
    #     save_name=synthetic_data_save_name
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
        trainer.load_fold_split(fold, from_save=False, save=True)
        
        start_time = time.perf_counter()
        # trainer.train(
        #     batch_size=batch_size,
        #     epoch=1000,
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
        synthetic_dataset = generator.generate_conditioned(labels)

        def pathfn(plot_type: str, label: tuple):
            prefix = os.path.join(generator.save_dir, 'plots')

            '_'.join(list(labels[label].values())) if label in labels else 'all_labels'
            label = label.replace('.', '').replace('/', '').replace(' ', '').lower()
            path_str = f"{prefix}_{plot_type}_{label}"

            return path_str

        # convert the data for standard format for downstream tasks
        evaluator = Evaluator(generator=generator, processor=processor)

        stats = evaluator.evaluate(
            source=trainer.train_dataset,
            synthetic=pickle.load(file=open(generator.save_path, 'rb')),
            get_plot_path_fn=pathfn,
            compare_label=age_label_compare_labels,
        )
        print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))

        break

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