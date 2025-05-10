import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class SleepQADataset(BaseDataset):
    
    df: pd.DataFrame
    """
    Output Dataset 
    

    References:
        - SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering
        -- Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Jaime Rei Ern Pang, Jessica Chen, 
        -- Vaaruni Nayak, Shafiq Joty, Josip Car Proceedings of the 2nd Machine Learning for Health symposium, 
        -- PMLR 193:199-217, 2022.
        -- https://proceedings.mlr.press/v193/bojic22a.html
        -- https://github.com/IvaBojic/SleepQA
        
    Data Fields:
        - q_p1  question asked to the LLMs(1,2)
        - par_1 - answer paragraph from LLM[1]
        - par_2 - answer paragraph from LLM[2]
        - answer_1 - short value answer from LLM[1]
        - answer_2 - short value answer from LLM[2]
        - score_a_1 - annotator 1 score for LLM[1] answer
        - score_p_1 - annotator 1 score for LLM[2] answer
        - score_a_2 - annotator 2 score for LLM[1] answer
        - score_p_2 - annotator 2 score for LLM[2] answer
        - score_a_3 - annotator 3 score for LLM[1] answer
        - score_p_3 - annotator 3 score for LLM[2] answer
        - score_a_4 - annotator 4 score for LLM[1] answer
        - score_p_4 - annotator 4 score for LLM[2] answer
        - score_a_5 - annotator 5 score for LLM[1] answer
        - score_p_5 - annotator 5 score for LLM[2] answer

    Functions:
        - SearchSleepQAQuestions: Task to Search for sleep-related questions in the dataset.
        - __init__: Initialize the SleepQADataset class.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            root (str): Root directory of the dataset.
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            config_path (str, optional): Path to the configuration file. Defaults to None.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "sleepQA.yaml"
            )
        default_tables = ["sleepqa_model_agreement"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "sleepqa_model_agreement",
            config_path=config_path,
        )

        self.df = pd.read_csv(str(root) + "\\model_agreement.csv")

        return
    
    def SearchSleepQAQuestions(self, search_string: str) -> pd.DataFrame:
        """
        SearchSleepQAQuestions is a class that represents a task for searching sleep-related questions in a dataset.
        
        This class is designed to work with the SleepQA dataset, which contains questions and answers related to sleep.
        
        Args:
            search_string (str): The string to search for in the dataset.
        
        Returns:
            datatable (pd.DataFrame): A DataFrame containing the search results.

        """
        task_name: str = "SearchSleepQAQuestions"
        input_schema: str
        output_schema: pd.DataFrame

        # Perform the search operation on the dataset
        columns_to_search = ['q_p1', 'par_1', 'par_2', 'answer_1', 'answer_2']
        mask = self.df[columns_to_search].apply(lambda x: x.str.contains(search_string, na=False)).any(axis=1)
        result_df = self.df[mask]
        # Return the resulting DataFrame
        return result_df.reset_index(drop=True)