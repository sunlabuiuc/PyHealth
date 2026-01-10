import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from pyhealth.datasets import DREAMTDataset
from pyhealth.datasets import DREAMTSleepWakeDataset
class TestDREAMTDatasetNewerVersions(unittest.TestCase):
    """Test DREAMT dataset containing 64Hz and 100Hz folders with local test data."""
    
    def setUp(self):
        """Set up participant info csv and 64Hz 100Hz files"""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        (self.root / "data_64Hz").mkdir()
        (self.root / "data_100Hz").mkdir()

        self.num_patients = 5
        patient_data = {
            'SID': [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            'AGE': np.random.uniform(25, 65, self.num_patients),
            'GENDER': np.random.choice(['M', 'F'], self.num_patients),
            'BMI': np.random.randint(20, 50, self.num_patients),
            'OAHI': np.random.randint(0, 50, self.num_patients),
            'AHI': np.random.randint(0, 50, self.num_patients),
            'Mean_SaO2': [f"{val}%" for val in np.random.randint(85, 99, self.num_patients)],
            'Arousal Index': np.random.randint(1, 100, self.num_patients),
            'MEDICAL_HISTORY': ['Medical History'] * self.num_patients,
            'Sleep_Disorders': ['Sleep Disorder'] * self.num_patients,
        }

        patient_data_df = pd.DataFrame(patient_data)
        patient_data_df.to_csv(self.root / "participant_info.csv", index=False)
        self._create_files()

    def _create_files(self):
        """Create 64Hz and 100Hz files"""
        for i in range(1, self.num_patients + 1):
            sid = f"S{i:03d}"
           
            partial_data = {
            'TIMESTAMP': [np.random.uniform(0, 100)],
            'BVP': [np.random.uniform(1, 10)],
            'HR': [np.random.randint(15, 100)],
            'EDA': [np.random.uniform(0, 1)],
            'TEMP': [np.random.uniform(20, 30)],
            'ACC_X': [np.random.uniform(1, 50)],
            'ACC_Y': [np.random.uniform(1, 50)],
            'ACC_Z': [np.random.uniform(1, 50)],
            'IBI': [np.random.uniform(0.6, 1.2)],
            'Sleep_Stage': [np.random.choice(['W', 'N1', 'N2', 'N3', 'R'])],
            }

            pd.DataFrame(partial_data).to_csv(self.root / "data_64Hz" / f"{sid}_whole_df.csv")
            pd.DataFrame(partial_data).to_csv(self.root / "data_100Hz" / f"{sid}_PSG_df.csv")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test DREAMTDataset initialization"""
        dataset = DREAMTDataset(root=str(self.root))
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        """Test dreamt-metadata.csv created"""
        dataset = DREAMTDataset(root=str(self.root))
        metadata_file = self.root / "dreamt-metadata.csv"
        self.assertTrue(metadata_file.exists())

    def test_patient_count(self):
        """Test all patients are added"""
        dataset = DREAMTDataset(root=str(self.root))
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)
    
    def test_stats_method(self):
        """Test stats method"""
        dataset = DREAMTDataset(root=str(self.root))
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = DREAMTDataset(root=str(self.root))
        patient = dataset.get_patient('S001')
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, 'S001')
    
    def test_get_patient_not_found(self):
        """Test that patient not found throws error."""
        dataset = DREAMTDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient('S222')
    
class TestDREAMTSleepWakeDataset(unittest.TestCase):
    """Test DREAMTSleepWakeDataset with minimal realistic sample data."""

    def setUp(self):
        """Set up temporary dataset_sample folder with participant info, features, and results."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create dataset_sample folder
        self.dataset_sample_dir = self.root / "dataset_sample"
        self.dataset_sample_dir.mkdir()

        # Create features_df folder inside dataset_sample
        self.features_dir = self.dataset_sample_dir / "features_df"
        self.features_dir.mkdir()

        # Create results folder 
        self.results_dir = self.root / "results"
        self.results_dir.mkdir()

        # Number of patients for the test
        self.num_patients = 2  # can be 2 for simplicity

        # Create participant_info.csv inside dataset_sample
        patient_data = {
            'SID': [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            'AGE': [30, 40],
            'GENDER': ['M', 'F'],
            'BMI': [25, 28],
            'OAHI': [5, 10],
            'AHI': [6, 12], 
            'Mean_SaO2': [95, 97],
            'Arousal Index': [15, 20],
            'MEDICAL_HISTORY': ['None', 'None'],
        }

        patient_data_df = pd.DataFrame(patient_data)
        patient_data_df['SID'] = patient_data_df['SID'].astype(str)
        patient_data_df.to_csv(self.dataset_sample_dir / "participant_info.csv", index=False)

        # Create example feature files for each patient in features_df
        feature_columns = [
            'PPG_Rate_Mean','HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD','HRV_CVNN','HRV_CVSD','HRV_MedianNN',
            'HRV_MadNN','HRV_MCVNN','HRV_IQRNN','HRV_SDRMSSD','HRV_Prc20NN','HRV_Prc80NN','HRV_pNN50','HRV_pNN20',
            'HRV_MinNN','HRV_MaxNN','HRV_HTI','HRV_TINN','HRV_LF','HRV_HF','HRV_VHF','HRV_TP','HRV_LFHF','HRV_LFn',
            'HRV_HFn','HRV_LnHF','HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S','HRV_CSI','HRV_CVI','HRV_CSI_Modified',
            'HRV_PIP','HRV_IALS','HRV_PSS','HRV_PAS','HRV_GI','HRV_SI','HRV_AI','HRV_PI','HRV_C1d','HRV_C1a','HRV_SD1d',
            'HRV_SD1a','HRV_C2d','HRV_C2a','HRV_SD2d','HRV_SD2a','HRV_Cd','HRV_Ca','HRV_SDNNd','HRV_SDNNa','HRV_DFA_alpha1',
            'HRV_MFDFA_alpha1_Width','HRV_MFDFA_alpha1_Peak','HRV_MFDFA_alpha1_Mean','HRV_MFDFA_alpha1_Max',
            'HRV_MFDFA_alpha1_Delta','HRV_MFDFA_alpha1_Asymmetry','HRV_MFDFA_alpha1_Fluctuation','HRV_MFDFA_alpha1_Increment',
            'HRV_ApEn','HRV_SampEn','HRV_ShanEn','HRV_FuzzyEn','HRV_MSEn','HRV_CMSEn','HRV_RCMSEn','HRV_CD','HRV_HFD',
            'HRV_KFD','HRV_LZC','total_power','normalized_power','LF_frequency_power','HF_frequency_power','LF_frequency_peak',
            'HF_frequency_peak','LF_normalized_power','HF_normalized_power','breathing_rate','HR_mean','HR_median','HR_max',
            'HR_min','HR_range','HR_std','BVP_mean','BVP_median','BVP_max','BVP_min','BVP_range','BVP_std',
            'ACC_X_trimmed_mean','ACC_X_trimmed_max','ACC_X_trimmed_IQR','ACC_Y_trimmed_mean','ACC_Y_trimmed_max',
            'ACC_Y_trimmed_IQR','ACC_Z_trimmed_mean','ACC_Z_trimmed_max','ACC_Z_trimmed_IQR','ACC_X_MAD_trimmed_mean',
            'ACC_X_MAD_trimmed_max','ACC_X_MAD_trimmed_IQR','ACC_Y_MAD_trimmed_mean','ACC_Y_MAD_trimmed_max',
            'ACC_Y_MAD_trimmed_IQR','ACC_Z_MAD_trimmed_mean','ACC_Z_MAD_trimmed_max','ACC_Z_MAD_trimmed_IQR','ACC_INDEX',
            'TEMP_mean','TEMP_median','TEMP_max','TEMP_min','TEMP_std','mean_SCR_Height','max_SCR_Height','mean_SCR_Amplitude',
            'max_SCR_Amplitude','mean_SCR_RiseTime','max_SCR_RiseTime','mean_SCR_RecoveryTime','max_SCR_RecoveryTime',
            'timestamp_start','circadian_cosine','circadian_decay','circadian_linear','Sleep_Stage','Obstructive_Apnea',
            'Central_Apnea','Hypopnea','Multiple_Events','artifact','sid',
        ]
        num_rows_per_patient = 5  # <-- at least 2 for gradient computation

        for i, sid in enumerate(patient_data['SID']):  # make sure we use SID in correct format
            # Generate random data
            df = pd.DataFrame(np.random.rand(num_rows_per_patient, len(feature_columns)), columns=feature_columns)
            
            # Set correct string SID
            df['sid'] = sid  # 'S001', 'S002', etc.
            
            # Randomly assign sleep stages
            df['Sleep_Stage'] = np.random.choice(['W', 'N1', 'N2', 'N3', 'R'], size=num_rows_per_patient)

            # Save to CSV
            df.to_csv(self.features_dir / f"{sid}_domain_features_df.csv", index=False)

        # Create example quality_scores_per_subject.csv in results folder
        quality_data = {
            'sid': [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            'total_segments': [1000, 1000],
            'num_excludes': [100, 50],
            'percentage_excludes': [0.1, 0.05],
        }
        quality_df = pd.DataFrame(quality_data)
        quality_df['sid'] = quality_df['sid'].astype(str)

        pd.DataFrame(quality_data).to_csv(self.results_dir / "quality_scores_per_subject.csv", index=False)
    
    def tearDown(self):
        """Remove temporary folder after test"""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly and sets key attributes."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        self.assertIsNotNone(dataset)
        self.assertTrue(hasattr(dataset, 'clean_df'))
        self.assertTrue(hasattr(dataset, 'new_features'))
        self.assertTrue(hasattr(dataset, 'good_quality_sids'))

    def test_clean_df_metadata_created(self):
        """Test that the clean dataframe metadata file is created."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        clean_df_file = self.root / "dreamt-metadata.csv"
        self.assertTrue(clean_df_file.exists())

    def test_patient_count(self):
        """Test that all patients are included in good_quality_sids."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        self.assertEqual(len(dataset.good_quality_sids), 2)
'''
    def test_stats_method(self):
        """Test that stats method runs without error."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        dataset.stats()

    def test_get_patient_method(self):
        """Test that get_patient returns a valid patient object."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        patient_id = dataset.good_quality_sids[0]
        patient = dataset.get_patient(patient_id)
        self.assertEqual(patient.patient_id, patient_id)

    def test_get_patient_not_found(self):
        """Test that get_patient raises an error for unknown patient."""
        dataset = DREAMTSleepWakeDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient("S999")
            '''

if __name__ == "__main__":
    #unittest.main()
    # running only the new dataset, please change this when merging
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestDREAMTSleepWakeDataset))