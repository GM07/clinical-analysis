from typing import List
import pandas as pd
from src.data.dataset import Dataset
from src.model_registry import FastModel, ModelRegistry
from datasets import Dataset as HuggingFaceDataset
import logging

from src.pipelines.dataset_inference_pipeline import HuggingFaceDatasetInferencePipeline

logger = logging.getLogger(__name__)

class EvaluatorDataset:

    def __init__(self, mimic_raw: str, mimic_path: str, tokenizer_path: str):
        self.mimic_raw = pd.read_csv(mimic_raw)
        self.mimic = pd.read_csv(mimic_path)
        self.tokenizer = ModelRegistry.load_single_tokenizer(tokenizer_path)

    def generate_dataset(self, valid_domains: List[str], max_token_length: int = 2048, max_nb_notes_per_domain: int = None):
        """
        Generates a dataset of notes from the raw data that are in the valid domains and have a token length less than or equal to the max token length

        Args: 
            valid_domains: List of valid domains to include in the dataset
            max_token_length: Maximum token length of the notes to include in the dataset
            max_nb_notes_per_domain: Maximum number of notes per domain to include in the dataset
        """

        # Assert that the valid domains are present in the raw data
        assert set(valid_domains).issubset(set(self.mimic_raw['CATEGORY'].unique())), "One or more of the domains provided are not present in the raw data"

        mimic_raw_filtered = self.mimic_raw.copy()

        # Filter the raw data to only include the valid domains
        mimic_raw_filtered = mimic_raw_filtered[mimic_raw_filtered['CATEGORY'].isin(valid_domains)]

        logger.info(f"Number of notes left after filtering by domain: {len(mimic_raw_filtered)}")

        # Remove notes that are present in the processed mimic dataset
        mimic_raw_filtered = mimic_raw_filtered[~mimic_raw_filtered['ROW_ID'].isin(self.mimic['ROW_ID'].tolist())]
        logger.info(f"Number of notes left after removing notes present in the processed mimic dataset: {len(mimic_raw_filtered)}")

        # Only keep the first max_nb_notes_per_domain notes per domain
        if max_nb_notes_per_domain is not None:
            mimic_raw_filtered = mimic_raw_filtered.groupby('CATEGORY').head(max_nb_notes_per_domain)

        logger.info(f"Number of notes left after filtering by domain and max number of notes per domain: {len(mimic_raw_filtered)}")

        # Compute length of the notes
        mimic_raw_filtered['TOKEN_LENGTH'] = mimic_raw_filtered['TEXT'].apply(lambda x: len(self.tokenizer.encode(x)))
        mimic_raw_filtered = mimic_raw_filtered[mimic_raw_filtered['TOKEN_LENGTH'] <= max_token_length]

        logger.info(f"Number of notes left after filtering by token length: {len(mimic_raw_filtered)}")
        
        return mimic_raw_filtered

class EvaluatorDatasetSummarizer:

    def __init__(self, dataset_path: str, model_checkpoint: str):
        self.dataset = HuggingFaceDataset.from_csv(dataset_path)
        self.pipeline = HuggingFaceDatasetInferencePipeline(model_checkpoint, input_column='CHAT', output_column='SUMMARY')

    def prepare_dataset(self, load_from_cache_file: bool = True):
        self.dataset = self.dataset.map(self.prepare_row, load_from_cache_file=load_from_cache_file, remove_columns=self.dataset.column_names)
        return self.dataset

    def prepare_row(self, row):
        return {'ROW_ID': row['ROW_ID']} | {'TEXT': row['TEXT']} | {'CHAT': [
            {
                'role': 'system', 
                'content': 'Your role is to summarize the clinical note provided by the user. Only output the summary, no other text.'},
            {
                'role': 'user',
                'content': f'{row["TEXT"]}'
            }
        ]}


    def summarize(self):
        """
        Summarizes the dataset using the pipeline

        Args:
            batch_size: Batch size for the inference pipeline
        """
        self.dataset = self.dataset.select(range(1000))
        self.dataset = self.pipeline(self.dataset)
        return self.dataset
