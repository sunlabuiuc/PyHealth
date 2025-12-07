import unittest
import os
import shutil
import numpy as np
import subprocess
import pickle
import sys
from pyedflib import highlevel

class TestTUHProcessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Setup temporary directories and generate mock data"""
        cls.base_dir = "./test_tuh_env"
        cls.train_dir = os.path.join(cls.base_dir, "mock_data", "train")
        cls.dev_dir = os.path.join(cls.base_dir, "mock_data", "dev")
        cls.output_dir = os.path.join(cls.base_dir, "output")
        cls.script_path = "process_TUH_dataset.py" 
        
        # Clean start (unless we are just inspecting existing data)
        if os.path.exists(cls.base_dir):
            shutil.rmtree(cls.base_dir)
        
        os.makedirs(cls.train_dir, exist_ok=True)
        os.makedirs(cls.dev_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create mock EDF and Annotation for both Train and Dev
        cls.create_mock_data(cls.train_dir, "patient_01_train")
        cls.create_mock_data(cls.dev_dir, "patient_02_dev")

    @classmethod
    def create_mock_data(cls, directory, filename_base):
        """Generates a dummy EDF file and a .tse_bi file in the specified directory"""
        # 1. Define Standard TUH Channels
        channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
                    'EEG F7-REF', 'EEG F8-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG CZ-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG P3-REF', 
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T5-REF', 
                    'EEG T6-REF', 'EEG PZ-REF', 'EEG FZ-REF']
        
        fs = 250 # Sample frequency (Hz)
        duration = 60 # seconds
        nsamples = duration * fs
        
        signals = []
        signal_headers = []
        
        for ch in channels:
            # Create random signal
            signals.append(np.random.rand(nsamples) * 100) 
            signal_headers.append({
                'label': ch, 
                'dimension': 'uV', 
                'sample_frequency': fs,
                'physical_min': -100, 
                'physical_max': 100, 
                'digital_min': -32768, 
                'digital_max': 32767
            })
            
        header = {'technician': 'test_tech', 'recording_additional': 'test_rec'}
        
        edf_filename = os.path.join(directory, f"{filename_base}.edf")
        highlevel.write_edf(edf_filename, signals, signal_headers, header)
        
        # 2. Create Annotation (.tse_bi)
        tse_content = [
            "version = tse_v1.0.0",
            "",
            "0.0000 10.0000 bckg 1.0000",
            "10.0000 45.0000 seiz 1.0000", # 35 seconds of seizure
            "45.0000 60.0000 bckg 1.0000"
        ]
        
        tse_filename = os.path.join(directory, f"{filename_base}.tse_bi")
        with open(tse_filename, "w") as f:
            f.write("\n".join(tse_content))

    def test_01_data_generation(self):
        """Test 1: Verify mock data exists and is valid"""
        edf_files_dev = [f for f in os.listdir(self.dev_dir) if f.endswith('.edf')]
        tse_files_train = [f for f in os.listdir(self.train_dir) if f.endswith('.tse_bi')]
        
        self.assertTrue(len(edf_files_dev) > 0, "No Dev EDF files generated")
        self.assertTrue(len(tse_files_train) > 0, "No Train Annotation files generated")
        print("\n[PASS] Test 1: Data Generation - Verified that mock EDF files and annotation (.tse_bi) files were successfully created in test directories")

    def test_02_pipeline_execution(self):
        """Test 2: Execute the script via subprocess and check exit code"""
        print("\n[INFO] Running processor script on mock dev data...")
        cmd = [
            "python", self.script_path,
            "--data_folder", self.dev_dir,
            "--data_type", "dev",
            "--save_directory", self.output_dir,
            "--samplerate", "200",
            "--label_type", "tse_bi",
            "--cpu_num", "1", 
            "--feature_sample_rate", "50",
            "--use_dev_function" 
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Always print output for debugging
        if result.stdout:
            print("Script stdout:")
            print(result.stdout)
        if result.stderr:
            print("Script stderr:")
            print(result.stderr)
        
        if result.returncode != 0:
            print("Script failed with error:")
            print(result.stderr)
            
        self.assertEqual(result.returncode, 0, "Script execution failed")
        print("\n[PASS] Test 2: Pipeline Execution - Successfully executed the processing script (process_TUH_dataset.py) on mock dev data without errors")

    def test_03_output_existence(self):
        """Test 3: Check if output structure is created and contains data"""
        expected_subdir = "dataset-tuh_task-binary_datatype-dev_v6"
        full_out_path = os.path.join(self.output_dir, expected_subdir)
        
        self.assertTrue(os.path.exists(full_out_path), 
                       f"Output subdirectory not created at {full_out_path}")
        
        # List all files in the directory for debugging
        all_files = os.listdir(full_out_path) if os.path.exists(full_out_path) else []
        pkl_files = [f for f in all_files if f.endswith('.pkl') and 'preprocess_info' not in f]
        
        self.assertTrue(len(pkl_files) > 0, 
                       f"No signal pickle files found in output. Directory: {full_out_path}, "
                       f"Files present: {all_files}")
        self.assertTrue(os.path.exists(os.path.join(full_out_path, "preprocess_info.infopkl")),
                       f"preprocess_info.infopkl not found in {full_out_path}")
        print(f"\n[PASS] Test 3: Output Existence - Verified that output directory structure was created correctly with {len(pkl_files)} signal pickle file(s) and preprocess_info.infopkl metadata file")

    def test_04_data_integrity(self):
        """Test 4: Load a pickle file and verify shape/content"""
        expected_subdir = "dataset-tuh_task-binary_datatype-dev_v6"
        full_out_path = os.path.join(self.output_dir, expected_subdir)
        pkl_files = [f for f in os.listdir(full_out_path) if f.endswith('.pkl') and 'preprocess_info' not in f]
        
        # Check if pickle files exist before trying to access them
        self.assertTrue(len(pkl_files) > 0, 
                       f"No signal pickle files found in {full_out_path}. Available files: {os.listdir(full_out_path)}")
        
        file_to_test = os.path.join(full_out_path, pkl_files[0])
        
        with open(file_to_test, 'rb') as f:
            data = pickle.load(f)
            
        self.assertIn('RAW_DATA', data)
        self.assertIn('LABEL1', data)
        
        raw_data = data['RAW_DATA'][0]
        
        # Check Raw Data Dimensions (19 Channels)
        self.assertEqual(raw_data.shape[0], 19, "Incorrect number of channels in output")
        self.assertTrue(raw_data.dtype == np.float16, "RAW_DATA is not float16")
        print(f"\n[PASS] Test 4: Data Integrity - Verified that processed pickle files contain valid data: RAW_DATA with 19 channels (shape: {raw_data.shape}), correct dtype (float16), and required keys (RAW_DATA, LABEL1)")

    @classmethod
    def tearDownClass(cls):
        """Cleanup temporary files"""
        # Only cleanup if NOT in generate-only mode
        if not os.environ.get("KEEP_DATA"):
            if os.path.exists(cls.base_dir):
                shutil.rmtree(cls.base_dir)

if __name__ == '__main__':
    if '--generate_mock_data' in sys.argv:
        # Generate Data Only Mode
        sys.argv.remove('--generate_mock_data') # clean args for any subprocess if needed
        os.environ["KEEP_DATA"] = "True"
        
        print("[INFO] Generating mock data...")
        TestTUHProcessing.setUpClass()
        print(f"[SUCCESS] Mock data generated at: {TestTUHProcessing.base_dir}")
        print(f"[INFO] Train dir: {TestTUHProcessing.train_dir}")
        print(f"[INFO] Dev dir:   {TestTUHProcessing.dev_dir}")
    else:
        # Run Full Test Suite
        unittest.main()