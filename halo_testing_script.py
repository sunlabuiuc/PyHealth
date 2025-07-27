print("BEGIN: Testing")
import subprocess, sys, os
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--force-reinstall"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "--force-reinstall"])
print("Success on pip install -e .")
from pyhealth.models.generators.halo import HALO
from pyhealth.models.generators.halo_resources.halo_model import HALOModel
from pyhealth.models.generators.halo_resources.halo_config import HALOConfig
from pyhealth.datasets.halo_mimic3 import HALO_MIMIC3Dataset
print("Sucess on imports")
print(f"Operating in dir: {os.getcwd()}")

halo_config = HALOConfig()
halo_dataset = HALO_MIMIC3Dataset(mimic3_dir="../../../../scratch_old/ethanmr3/mimic3/physionet.org/files/mimiciii/1.4/", pkl_data_dir="../../halo_pkl/", gzip=True)
model = HALO(dataset=halo_dataset, config=halo_config, save_dir="../../halo_save/", train_on_init=False)
print("Success on model setup")

model.train()
print("Sucess on model train")

model.test(testing_results_dir = "../../halo_results/")
print("Success on model test")

model.synthesize_dataset(pkl_save_dir = "../../halo_results/")
print("Success on dataset synthesis")

print("END: Testing success!!!")