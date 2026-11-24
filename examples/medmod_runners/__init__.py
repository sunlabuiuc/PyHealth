"""Scripts that produced the real-data results of the MedMod examples.

These are the training and evaluation scripts we ran on Google Colab, as shown
in ``examples/mimic4_medmod_ihm_colab_runs.ipynb``. They train the same models
with the same settings as the example scripts in ``examples/``. They add
bootstrap confidence intervals and a JSON report per run that records the code
version (the git commit, or the time the Colab code zip was made) and the
SHA-256 of the script files. Each runner saves checkpoints, the Early Fusion
and probe runners every 10 epochs and the ConVIRT pretraining runner after
every epoch. Running the same command again continues an interrupted run. The
results page
``examples/mimic4_medmod_ihm_results.md`` explains the settings and deviations.
"""
