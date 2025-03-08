from typing import List
import uuid
from src.data.dataset import VerbalizedExtractionDataset

from datasets import Dataset as HuggingFaceDataset, concatenate_datasets
import pandas as pd

class HumanEvaluation:

    """
    In order to perform human evaluation, we need to have a dataset with the following columns:

    - id: Unique identifier for the sample generated
    - hadm_id: Unique identifier for the admission in the MIMIC-III dataset
    - clinical_notes : Clinical notes that were used to generate the summary
    - summary: Summary that was generated by the model
    - expected_domain : The domain that the summary is expected to be relevant to
    - relevance : Score (1-10) which indicates how relevant the summary is to the expected domain
        1 being that the summary is not relevant to the expected domain at all
        10 being that all the information in the summary is relevant to the expected domain
    - groundedness : Score (1-10) which indicates how grounded the summary is
        1 being that the summary is not grounded in the clinical notes at all
        10 being that the summary is fully grounded in the clinical notes
    - recall : Score (1-10) which indicates how much of the information in the clinical notes is present in the summary 
        1 being that the summary contains none of the information in the clinical notes
        10 being that the summary contains all the information in the clinical notes that is relevant to the expected domain
    - method : Which method was used to generate the summary
    - coherence : Score (1-10) which indicates how coherent the summary is
        1 being that the summary is not coherent at all
        10 being that the summary is fully coherent
    - fluency : Score (1-10) which indicates how fluent the summary is
        1 being that the summary is not fluent at all (bullet points, etc.)
        10 being that the summary is fully fluent (no bullet points, etc.)
    """


    def __init__(self, mimic_path: str):
        self.mimic_path = mimic_path
        self.dataset = pd.read_csv(mimic_path)

        self.create_index()

    def create_index(self):
        self.dataset = self.dataset[self.dataset['CATEGORY'] != 'Discharge summary']

        def admission_to_prompt(clinical_notes_series):
            clinical_notes = clinical_notes_series.tolist()

            clinical_note_string = ''
            for i, note in enumerate(clinical_notes):
                clinical_note_string += f'Clinical note {i+1}:\n{note}\n'

            return clinical_note_string

        self.id_to_notes = self.dataset.groupby('HADM_ID')['TEXT'].aggregate(admission_to_prompt)

    def from_datasets(
        self, 
        verbalized_datasets_paths: List[str], 
        verbalized_columns: List[str], 
        baseline_datasets_paths: List[str], 
        output_path: str = None, 
        max_samples_per_dataset: int = 10,
        verbalized_methods: List[str] = None,
        baseline_methods: List[str] = None
    ):
        """
        Converts a list of verbalized datasets and a list of baseline datasets to a human evaluation dataset

        Args:
            verbalized_datasets_paths: List of paths to the verbalized datasets
            verbalized_columns: List of columns to use from the verbalized datasets
            baseline_datasets_paths: List of paths to the baseline datasets
            output_path: Path to save the dataset
            max_samples_per_dataset: Maximum number of samples to convert per dataset
            verbalized_methods: List of methods to use for the verbalized datasets (must be same length as verbalized_datasets_paths)
            baseline_methods: List of methods to use for the baseline datasets (must be same length as baseline_datasets_paths)
        """

        datasets = []
        
        if verbalized_methods is None:
            verbalized_methods = [''] * len(verbalized_datasets_paths)
        else:
            assert len(verbalized_methods) == len(verbalized_datasets_paths), 'verbalized_methods must be the same length as verbalized_datasets_paths'

        if baseline_methods is None:
            baseline_methods = [''] * len(baseline_datasets_paths)
        else:
            assert len(baseline_methods) == len(baseline_datasets_paths), 'baseline_methods must be the same length as baseline_datasets_paths'

        for verbalized_dataset_path, method in zip(verbalized_datasets_paths, verbalized_methods):
            dataset = self.from_verbalized_dataset(verbalized_dataset_path, verbalized_columns, method=method, output_path=None, max_samples=max_samples_per_dataset)
            datasets.append(dataset)

        for baseline_dataset_path, method in zip(baseline_datasets_paths, baseline_methods):
            dataset = self.from_baseline_dataset(baseline_dataset_path, method=method, output_path=None, max_samples=max_samples_per_dataset)
            datasets.append(dataset)

        final_dataset = concatenate_datasets(datasets)
        final_dataset = final_dataset.map(lambda x: {'length': len(x['summary'])}).filter(lambda x: x['length'] > 200)
        final_dataset = final_dataset.remove_columns(['length'])
        final_dataset: HuggingFaceDataset = final_dataset.shuffle(seed=42)


        if output_path:
            final_dataset.to_csv(output_path)

        ready_for_evaluation = final_dataset.remove_columns(['method', 'hadm_id'])

        return final_dataset, ready_for_evaluation

    def from_verbalized_dataset(
        self, 
        dataset_path: str, 
        columns: List[str],
        method: str, 
        output_path: str = None, 
        max_samples: int = 10
    ):
        """
        Converts a verbalized dataset to a human evaluation dataset 

        Args:
            dataset_path: Path to the dataset
            method: Method that was used to generate the summary
            output_path: Path to save the dataset
            max_samples: Maximum number of samples to convert
        """


        # Trick to prevent loading the dataset since we only want the processing on the columns
        col_to_domains = dict(zip(columns, map(lambda x: '_'.join(x.split('_')[1:-1]), columns)))
        domains_to_col = dict(zip(map(lambda x: '_'.join(x.split('_')[1:-1]), columns), columns))

        dataset = VerbalizedExtractionDataset(columns=columns, dataset_path=dataset_path)
        data = dataset.filter_non_valid_generations()

        dataset = HuggingFaceDataset.from_pandas(data)

        def to_format(x):
            nb_new_samples = len(domains_to_col)
            hadm_id = x['HADM_ID'][0]
            return {
                'id': [str(uuid.uuid4()) for _ in range(nb_new_samples)],
                'hadm_id': [hadm_id] * nb_new_samples,
                'clinical_notes': [self.id_to_notes[hadm_id]] * nb_new_samples,
                'summary': [x[col][0] for col in columns],
                'expected_domain': [col_to_domains[col] for col in columns],
                'method': [method] * nb_new_samples,
                'relevance': [0] * nb_new_samples,
                'groundedness': [0] * nb_new_samples,
                'recall': [0] * nb_new_samples,
                'coherence': [0] * nb_new_samples,
                'fluency': [0] * nb_new_samples,
            }

        dataset = dataset.map(to_format, batched=True, batch_size=1, desc='Converting to human evaluation format', remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: x['summary'] is not None, desc='Filtering out empty summaries')
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(max_samples))

        if output_path:
            dataset.to_csv(output_path)

        return dataset

    def from_baseline_dataset(
        self, 
        dataset_path: str, 
        method: str, 
        output_path: str = None, 
        summary_column_name: str = 'OUTPUT', 
        max_samples: int = 10
    ):
        """
        Converts a baseline dataset to a human evaluation dataset 

        Args:
            dataset_path: Path to the dataset
            method: Method that was used to generate the summary
            output_path: Path to save the dataset
            summary_column_name: Name of the column that contains the summary
            max_samples: Maximum number of samples to convert
        """

        dataset = HuggingFaceDataset.from_csv(dataset_path)
        dataset = dataset.filter(lambda x: 'there is' not in x[summary_column_name])

        def to_format(x):
            nb_samples = len(x['HADM_ID'])
            return {
                'id': [str(uuid.uuid4()) for _ in range(nb_samples)],
                'hadm_id': x['HADM_ID'],
                'clinical_notes': self.id_to_notes[x['HADM_ID']],
                'summary': x[summary_column_name],
                'expected_domain': x['CATEGORY'],
                'method': [method] * nb_samples,
                'relevance': [0] * nb_samples,
                'groundedness': [0] * nb_samples,
                'recall': [0] * nb_samples,
                'coherence': [0] * nb_samples,
                'fluency': [0] * nb_samples,
            }

        dataset = dataset.map(to_format, batched=True, desc='Converting to human evaluation format', remove_columns=dataset.column_names)
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(max_samples))

        if output_path:
            dataset.to_csv(output_path)

        return dataset

    def from_dataset(self, dataset_path: str, output_path: str):
        pass
