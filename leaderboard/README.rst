PyHealth Leaderboard Generation
===============================

We provide a leaderboard generation functionality which periodically (weekly) runs each model in our package
on each task and dataset we have (i.e. run our pipeline with each dataset/task/model).
The result would be stored both locally and remotely
(to our `Google Spreadsheet <https://docs.google.com/spreadsheets/d/1c4OwCSDaEt7vGmocidq1hK2HCTeB6ZHDzAZvlubpi08/edit#gid=1602645797>`_ by default).

The aims/expectations of this leaderboard are:

* Regularly check the validity of the committed code -> better **Maintenance** of PyHealth
* Provide benchmarks for the performance of different healthcare machine learning models on different datasets/tasks \
  so that:
    1) Researchers can easily compare their models with others in the same setting.
    2) Anyone from the healthcare community can reproduce others' works easily.

To run this leaderboard generation and upload data to your own google spreadsheet, you need to download your GCP credential file (.json) and run the following
command line in your terminal:

.. code-block::

    cd ./leaderboard
    python3 leaderboard_gen.py \
    --credentials [YOUR CREDENTIAL FILE] \
    --doc_name [YOUR GOOGLE DOC FILE NAME] \
    --sheet_id [THE ID OF THE SHEET YOU WANT TO PUT DATA] \
    --out_path [LOCAL LOG PATH]

For more details regarding the connection to a remote Google Spreadsheet, you may refer to an online tutorial
`here <https://www.makeuseof.com/tag/read-write-google-sheets-python/>`_

If you want to manually update our spreadsheet instead of your owns', you should request a credential from us.

If you only want to **locally** generate the leaderboard, you may use the following command:

.. code-block::

    cd ./leaderboard
    python3 leaderboard_gen.py  --remote False


The results would be stored remotely to the google spreadsheet in the form like:

===========     =============    ============      ============      ============
Model Name      Jaccard	         Accuracy	       Macro-F1	         PRAUC
===========     =============    ============      ============      ============
LR              0.4284034819	 0.8906031666	   0.7617138219	     0.7396402706
RF              0.3789186834	 0.8841111372	   0.7360749644	     0.7087553597
NN              0.4155517272	 0.8821072015	   0.7533514874	     0.6984982175
RNN             0.4455980117	 0.8948890614	   0.7695140673	     0.7454222703
CNN             0.4327503087	 0.8872199455	   0.7612806993	     0.7344320306
Transformer     0.4306627263	 0.8875918471	   0.7597112762	     0.7357706761
RETAIN          0.4619954769	 0.8933003557	   0.7767088880	     0.7524748236
GAMENet         0.4387502568	 0.8892383239	   0.7638512419	     0.7166323735
MICRON          0.3584965503	 0.8806773664	   0.7245369254	     0.6985561461
SafeDrug        0.4335727149	 0.8856709574	   0.7610277014	     0.6469623656
===========     =============    ============      ============      ============

**If you commit your model to our package, the performance of your model would be shown on our leaderboard \
in the next week.**