pyhealth.tasks.SleepStagingDREAMT
=================================

Sleep staging preprocessing on DREAMT dataset, as described in Wang et al. 

Data is divided into 30-second epochs. Using blood volume pulse signals, the 
task calculates inter-beat intervals (IBI) values per epoch, removing anomalous
IBI values greater than 2. 

Paper:
    Wang et al. "WatchSleepNet: A Novel Model and Pretraining Approach for
    Advancing Sleep Staging with Smartwatches." CHIL 2025 (PMLR 287).

.. autoclass:: pyhealth.tasks.SleepStagingDREAMT   
   :members:
   :undoc-members:
   :show-inheritance:
