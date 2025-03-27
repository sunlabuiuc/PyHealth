import pyhealth.datasets.mimic3 as mimic3
import pyhealth.datasets.mimic4 as mimic4
import pyhealth.tasks.medical_coding as coding
import time

def time_function(func, name):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{name} execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # print("Starting MIMIC-III processing...")
    # time_function(mimic3.main, "MIMIC-III")
    # print("\nStarting MIMIC-IV processing...")
    # time_function(mimic4.main, "MIMIC-IV")
    print("\nStart Medical Coding Test")
    time_function(coding.main, "Medical Coding")