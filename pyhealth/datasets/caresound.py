"""CaReSound dataset for PyHealth.

This module provides the CaReSoundDataset class for loading and processing
the CaReSound benchmark data for Audio Question Answering (AQA) tasks.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class CaReSoundDataset(BaseDataset):
    """CaReSound dataset for open-ended diagnostic reasoning.

    This dataset aggregates five medical audio sources: ICBHI, KAUH, 
    CirCor, SPRSound, and ZCHSound. It pairs respiratory and cardiac 
    audio with 34,792 GPT-4o generated Question-Answer pairs.

    Args:
        root: Root directory containing the audio files (.wav) and/or CSVs.
        tables: Optional list of tables to load. Defaults to ["metadata"].
        dataset_name: Optional name of the dataset. Defaults to "caresound".
        config_path: Optional path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "caresound.yaml"

        # 1. Prepare standardized CSV (handles all local/API edge cases)
        pyhealth_csv = os.path.join(root, "caresound_metadata.csv")
        if not os.path.exists(pyhealth_csv):
            self.prepare_metadata(root)

        # 2. Resolve local audio paths dynamically
        self.audio_path_map = self._resolve_audio_paths(root)

        # 3. Define the default table mapped in the YAML
        default_tables = ["metadata"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "caresound",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepares QA metadata from local ZIP/CSV drops or downloads via HF API."""
        output_path = os.path.join(root, "caresound_metadata.csv")
        
        train_csv = os.path.join(root, "CaReSoundQA_train.csv")
        test_csv = os.path.join(root, "CaReSoundQA_test.csv")
        full_csv = os.path.join(root, "CaReSoundQA.csv")
        
        df_master = None
        
        # Scenario A: User manually downloaded both Train and Test CSVs
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            logger.info("Found local train/test CSVs. Merging...")
            df_train, df_test = pd.read_csv(train_csv), pd.read_csv(test_csv)
            df_train['hf_split'], df_test['hf_split'] = 'train', 'test'
            df_master = pd.concat([df_train, df_test], ignore_index=True)

        # Scenario B: User manually downloaded ONLY Train CSV
        elif os.path.exists(train_csv):
            logger.warning("Found local train CSV, but test is missing. Using train only.")
            df_master = pd.read_csv(train_csv)
            df_master['hf_split'] = 'train'

        # Scenario C: User manually downloaded the Master CSV
        elif os.path.exists(full_csv):
            logger.info(f"Found master CSV: {full_csv}.")
            df_master = pd.read_csv(full_csv)
            if 'hf_split' not in df_master.columns:
                df_master['hf_split'] = 'unknown'

        # Scenario D: Fallback to Hugging Face API
        else:
            try:
                from datasets import load_dataset
                logger.info("Local metadata not found. Fetching from tsnngw/CaReSound...")
                dataset = load_dataset("tsnngw/CaReSound")
                
                df_train = dataset['train'].to_pandas()
                df_train['hf_split'] = 'train'
                
                # Catch in case the dataset structure changes on HF
                if 'test' in dataset:
                    df_test = dataset['test'].to_pandas()
                    df_test['hf_split'] = 'test'
                    df_master = pd.concat([df_train, df_test], ignore_index=True)
                else:
                    df_master = df_train
                
            except ImportError:
                logger.error("The 'datasets' library is required. Run: pip install datasets")
                raise
            except Exception as e:
                logger.error(f"Failed to fetch metadata: {e}")
                raise

        # ---> MINIMAL FIX: Inject audio paths right before saving <---
        audio_map = {}
        for path in Path(root).rglob("*.wav"):
            stem, path_str = path.stem, str(path).lower()
            source = "Unknown"
            if "icbhi" in path_str: source = "ICBHI"
            elif "circor" in path_str: source = "CirCor"
            elif "kauh" in path_str: source = "KAUH"
            elif "spr" in path_str: source = "SPRSound"
            elif "zch" in path_str: source = "ZCHSound"
            
            audio_map[(source, stem)] = str(path.absolute())
            audio_map[(source, stem.split('_')[0])] = str(path.absolute())

        df_master['audio_path'] = df_master.apply(
            lambda r: audio_map.get((str(r.get('dataset', 'Unknown')), str(r.get('patient_id', ''))), ""), 
            axis=1
        )

        # Save the final CSV for the new PyHealth Engine to pick up automatically
        df_master.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_master)} QA pairs with mapped audio to {output_path}")

    def _resolve_audio_paths(self, root: str) -> Dict[tuple, str]:
        """Maps .wav files using robust stem and prefix mapping."""
        audio_map = {}
        wav_files = list(Path(root).rglob("*.wav"))
        
        if not wav_files:
            logger.warning(f"No .wav files found in {root}.")
            return audio_map

        for path in wav_files:
            stem = path.stem
            path_str = str(path).lower()
            
            source = "Unknown"
            if "icbhi" in path_str: source = "ICBHI"
            elif "circor" in path_str: source = "CirCor"
            elif "kauh" in path_str: source = "KAUH"
            elif "spr" in path_str: source = "SPRSound"
            elif "zch" in path_str: source = "ZCHSound"
                
            # 1. Primary Mapping: Exact filename match
            audio_map[(source, stem)] = str(path.absolute())
            
            # 2. Fallback Mapping: Base Patient ID match (e.g., '101' from '101_1b1')
            base_id = stem.split('_')[0]
            if (source, base_id) not in audio_map:
                audio_map[(source, base_id)] = str(path.absolute())
            
        return audio_map

    def parse_func(self) -> Dict[str, Any]:
        """Merges tabular QA metadata with local audio paths."""
        csv_path = os.path.join(self.root, "caresound_metadata.csv")
        df = pd.read_csv(csv_path)
        
        patients = {}
        missing_sources = set()
        
        for _, row in df.iterrows():
            pid = str(row['patient_id'])
            source = str(row['dataset'])
            
            audio_path = self.audio_path_map.get((source, pid))
            
            if not audio_path:
                missing_sources.add(source)
                continue
                
            if pid not in patients:
                patients[pid] = {"patient_id": pid, "visits": {}}
            
            visit_id = f"{source}_{pid}"
            if visit_id not in patients[pid]["visits"]:
                patients[pid]["visits"][visit_id] = {
                    "visit_id": visit_id,
                    "audio_path": audio_path,
                    "events": []
                }
                
            patients[pid]["visits"][visit_id]["events"].append({
                "question": row.get('question', ''),
                "answer": row.get('answer', ''),
                "hf_split": row.get('hf_split', 'unknown')
            })

        if missing_sources:
            logger.warning(
                f"Audio files missing for datasets: {', '.join(missing_sources)}. "
                "Only available multi-modal samples have been loaded."
            )

        return patients

    @property
    def default_task(self):
        from pyhealth.tasks import CaReSoundAQA
        return CaReSoundAQA()