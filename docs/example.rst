Examples
========


Quick Start for Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We propose the idea of standard template, a formalized schema for healthcare datasets.
Ideally, as long as the data is scanned as the template we defined, the downstream
task processing and the use of ML models will be easy and standard. In short, it has the following structure:
**add a figure here**. The dataloader for different datasets can be found in examples/data_generation.
Using `"examples/data_generation/dataloader_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/dataloader_mimic_demo_parallel.py>`_
as an exmaple:

#. First read in patient, admission, and event tables.

   .. code-block:: python


       from pyhealth.utils.utility import read_csv_to_df
       patient_df = read_csv_to_df(os.path.join('data', 'mimic-iii-clinical-database-demo-1.4', 'PATIENTS.csv'))
       admission_df = read_csv_to_df(os.path.join('data', 'mimic-iii-clinical-database-demo-1.4', 'ADMISSIONS.csv'))
       ...

#. Then invoke the parallel program to parse the tables in n_jobs cores.

   .. code-block:: python


       from pyhealth.data.base_mimic import parallel_parse_tables
       all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
       delayed(parallel_parse_tables)(
            patient_df=patient_df,
            admission_df=admission_df,
            icu_df=icu_df,
            event_df=event_df,
            event_mapping_df=event_mapping_df,
            duration=duration,
            save_dir=save_dir)
        for i in range(n_jobs))

#. The processed sequential data will be saved in the prespecified directory.

   .. code-block:: python

      with open(patient_data_loc, 'w') as outfile:
          json.dump(patient_data_list, outfile)

The provided examples in PyHealth mainly focus on scanning the data tables in the schema we have, and **generate episode datasets**.
For instance, `"examples/data_generation/dataloader_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/dataloader_mimic_demo_parallel.py>`_
demonstrates the basic procedure of processing MIMIC III demo datasets.

#. The next step is to generate episode/sequence data for mortality prediction. See `"examples/data_generation/generate_mortality_prediction_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/generate_mortality_prediction_mimic_demo.py>`_

   .. code-block:: python

      with open(patient_data_loc, 'w') as outfile:
          json.dump(patient_data_list, outfile)

By this step, the dataset has been processed for generating X, y for phenotyping prediction. **It is noted that the API across most datasets are similar**.
One may easily replicate this procedure by calling the data generation scripts in \\examples\\data_generation. You may also modify the parameters in the
scripts to generate the customized datasets.

**Preprocessed datasets are also available at \\datasets\\cms and \\datasets\\mimic**.


----


Quick Start for Running Predictive Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running examples, you need the datasets. Please download from the GitHub repository `"datasets" <https://github.com/yzhao062/PyHealth/tree/master/datasets>`_.
You can either unzip them manually or running our script `"00_extract_data_run_before_learning.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/learning_models/00_extract_data_run_before_learning.py>`_

`"examples/learning_models/example_sequence_gpu_mortality.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/learning_models/example_sequence_gpu_mortality.py>`_
demonstrates the basic API of using GRU for mortality prediction. **It is noted that the API across all other algorithms are consistent/similar**.

**If you do not have the preprocessed datasets yet, download the \\datasets folder (cms.zip and mimic.zip) from PyHealth repository, and run \\examples\\learning_models\\extract_data_run_before_learning.py to prepare/unzip the datasets.**


#. Setup the datasets. X and y should be in x_data and y_data, respectively.

   .. code-block:: python

      # load pre-processed CMS dataset
      from pyhealth.data.expdata_generator import sequencedata as expdata_generator

      expdata_id = '2020.0810.data.mortality.mimic'
      cur_dataset = expdata_generator(exp_id=exp_id)
      cur_dataset.get_exp_data(sel_task='mortality', )
      cur_dataset.load_exp_data()


#. Initialize a LSTM model, you may set up the parameters of the LSTM, e.g., n_epoch, learning_rate, etc,.

   .. code-block:: python

      # initialize the model for training
      from pyhealth.models.sequence.lstm import LSTM
      # enable GPU
      clf = LSTM(expmodel_id=expmodel_id, n_batchsize=20, use_gpu=True,
          n_epoch=100, gpu_ids='0,1')
      clf.fit(cur_dataset.train, cur_dataset.valid)

#. Load the best shot of the training, predict on the test datasets

   .. code-block:: python

      # load the best model for inference
      clf.load_model()
      clf.inference(cur_dataset.test)
      pred_results = clf.get_results()


#. Evaluation on the model. Multiple metrics are supported.

   .. code-block:: python

      # evaluate the model
      from pyhealth.evaluation.evaluator import func
      r = func(pred_results['hat_y'], pred_results['y'])
      print(r)

