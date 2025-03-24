import logging
import os
import random
from typing import Any, List, Tuple
import joblib
import ast

from colorist import Color

import pandas as pd

logger = logging.getLogger(__name__)


class Dataset:
    """
    Dataset class used to load datasets, partition them between jobs and regroup partitions.
    This is only a pre-processing and post-processing class. It should not be used during inference
    """

    def __init__(self, dataset_path: str = None, data: pd.DataFrame = None):
        """
        Args:
            dataset_path: Path to dataset. For now, only csv files are supported
        """

        assert data is not None or dataset_path is not None, 'One of the arguments `dataset_path` or `data` must be provided'
        assert data is None or dataset_path is None, 'Only one of the arguments `dataset_path` or `data` must be provided'

        self.dataset_path = dataset_path

        if dataset_path is not None:
            self.load()
        else:
            self.data = data
    

    def load(self):
        """
        Loads the dataset from the `dataset_path` attribute
        """
        self.dataset_name = os.path.basename(self.dataset_path)
        self.dataset_folder_path = self.dataset_path.replace(self.dataset_name, '')

        # self.data = load_dataset('csv', data_dir=self.dataset_folder_path, data_files={'data': self.dataset_name})['data']
        self.data = pd.read_csv(self.dataset_path)

    def save(self, output_path: str = None):
        if output_path is None:
            output_path = self.dataset_path
        self.data.to_csv(output_path, index=False)

    def partition(self, output_folder_path: str, nb_partitions: int = None, size_of_partition: int = None, max_rows: int = None, overwrite: bool = False, original_dataset_path: str = None):
        """
        Partitions a dataset into multiple partitions (or shards) that can be used by different jobs.

        Args:
            output_folder_path: Path where the partition will be stored
            nb_partitions: Number of partitions to generate
            size_of_partition: Size of a single partition. If provided, will overide `nb_partitions`
            max_rows: Number of rows to consider when generating the partitions
            overwrite: Whether to overwrite a partition if a partition is already present
            original_dataset_path: Path to the original dataset. Needed if the dataset is not loaded from a file
        """
        assert nb_partitions is None or size_of_partition is None, f"One of the arguments `nb_partitions` or `size_of_partition` must be null"
        assert nb_partitions is not None or size_of_partition is not None, f"One of the arguments `nb_partitions` or `size_of_partition` must be provided"

        assert original_dataset_path is not None or self.dataset_path is not None, 'The original dataset path must be provided'
        dataset_path = original_dataset_path if original_dataset_path is not None else self.dataset_path

        # Partition handling
        if size_of_partition is not None:
            nb_partitions = (len(self.data) // size_of_partition) + 1
        else:
            size_of_partition = len(self.data) // nb_partitions

        # Folder handling
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
        
        if output_folder_path[-1] != '/':
            output_folder_path += '/'

        # Creating partitions
        current_rows_processed = 0
        for i in range(nb_partitions):
            start = i * size_of_partition
            end = min(len(self.data), (i + 1) * size_of_partition)

            partition = DatasetPartition(dataset_path, start, end, saving_path=output_folder_path + f'partition_{start}_{end}.partition')
            partition.save(overwrite=overwrite)

            current_rows_processed += end - start

            if max_rows is not None and current_rows_processed > max_rows:
                break

    @staticmethod
    def partitions_to_file(partition_folder_path: str, output_file_path: str, column_names: List[str] = ['normal', 'beam', 'constrained']):
        """
        Will create a csv file by merging the results of all partitions in `partition_folder_path`

        Args:
            partition_folder_path: Path of the folder containing all partitions
            output_file_path: Path where the output dataset will be saved
            column_names: Name of the columns where the results will be stored
        """
        analyzer = DatasetPartitionAnalyzer(partition_folder_path=partition_folder_path)

        if len(analyzer.partitions) == 0:
            return pd.DataFrame([])

        initial_dataset = pd.read_csv(analyzer.partitions[0].original_dataset_path)
        # nb_results = len(analyzer.partitions[0].results[0])
        # assert nb_results == len(column_names), 'The column names are not equal to the number of results per sample'
        if isinstance(analyzer.partitions[0].results[0], list) or isinstance(analyzer.partitions[0].results[0], tuple):
            nb_results = len(analyzer.partitions[0].results[0])
        else:
            nb_results = 1
        assert nb_results == len(column_names), f'The column names are not equal to the number of results per sample'
            
        for column_name in column_names:
            initial_dataset[f'{column_name}'] = None

        for partition in analyzer.partitions:
            for i, result_val in partition.results.items():
                if result_val is None:
                    continue
                    
                if isinstance(result_val, list) or isinstance(result_val, tuple):
                    for res, column_name in zip(result_val, column_names):
                        initial_dataset[column_name].iloc[partition.start + i] = res
                else:
                    initial_dataset[column_names[0]].iloc[partition.start + i] = result_val
        
        initial_dataset = initial_dataset.loc[:, ~initial_dataset.columns.str.contains('^Unnamed')]
        initial_dataset.to_csv(output_file_path, index=False)
        return initial_dataset

class DatasetPartition:
    """
    Class used to partition a dataset into multiple partitions (chunks) to allow jobs to run in parallel.

    This assumes a .csv file was used as the original dataset file
    """

    def __init__(self, original_dataset_path: str, start: int, end: int, saving_path: str = None):
        """
        Args:
            original_dataset_path: Path to original dataset used to generate this partition
            start: Id where the partition starts (.iloc[start])
            end: Id where the partition ends (.iloc[end])
            saving_path: Where the partition will be stored locally
        """
        self.original_dataset_path = original_dataset_path
        self.start = start
        self.end = end
        self.saving_path = saving_path

        self.nb_elements = end - start
        self.results = {i: None for i in range(self.nb_elements)}
        self.data = None

    def save_results(self, results: List[Tuple[int, Any]]):
        """
        Adds results to be stored with the partition and saves them with the partition

        Args:
            results: List of results to add to the current results
        """
        for id, value in results:
            self.results[id] = value 

        tmp_data = self.data
        self.save(overwrite=True)
        self.data = tmp_data

    def load(self):
        """
        Loads the dataset from the `original_dataset_path` path
        """
        self.data = pd.read_csv(self.original_dataset_path)

    def save(self, saving_path: str = None, overwrite: bool = True):
        """
        Saves the partition into a single file

        Args:
            saving_path: Path where the partition will be saved. This will override `self.saving_path` if provided
            overwrite: Whether to overwrite a partition if it already exists
        """
        if saving_path is None:
            saving_path = self.saving_path

        if not overwrite and os.path.exists(saving_path):
            logger.info(f'Partition already exists at {saving_path}')
            return

        self.data = None
        joblib.dump(self, saving_path)

    def is_completed(self):
        """
        Returns whether the partition was fully processed or not. 
        """
        return all(map(lambda x: x is not None, self.results.values()))

    def is_empty(self):
        return all(map(lambda x: x is None, self.results.values()))

    def nb_elements_unprocessed(self):
        """
        Returns the number of samples in the partition that are left to process
        """
        return len(list(filter(lambda x: x is None, self.results.values())))

    def nb_elements_processed(self):
        """
        Returns the number of samples in the partition that are processed
        """
        return len(list(filter(lambda x: x is not None, self.results.values())))

    def get_unprocessed_ids(self):
        return list(map(lambda x: x[0], filter(lambda x: x[1] is None, self.results.items())))

    def iterate(self):
        unprocessed_ids = self.get_unprocessed_ids()

        if len(unprocessed_ids) == 0:
            logger.warning('You are trying to process a partition that already has been processed entirely')
            return

        for i in self.get_unprocessed_ids():
            yield i, self[i]

    def clear_results(self):
        """
        Clear the results in the partition
        """
        self.results = {i: None for i in range(self.nb_elements)}

    def __getitem__(self, i: int):
        """
        Returns the ith element in the partition. The ith element is relative to the start. Thus, for i = 2, the element `start + 2` will be
        returned in the original dataset
        """
        assert self.data is not None, 'The method load() must be called prior to retrieving an item'
        assert i < self.nb_elements, f'The index {i} is invalid for a partition that only has {self.nb_elements} elements'

        return self.data.iloc[self.start + i]

    @classmethod
    def from_save(self, path: str, load_original_dataset: bool = True):
        """
        Loads a partition from a file

        Args:
            path: Path where to load the partition from
            load_original_dataset: Whether to load the original dataset with the partition
        """
        partition: DatasetPartition = joblib.load(path)
        partition.saving_path = path
        if load_original_dataset:
            partition.load()
        return partition

class DatasetPartitionAnalyzer:
    """
    Analyzes in a directory of dataset partitions which partitions are processed or not. This class can also dispatch partitions
    to different jobs by returning k partition files that are not done being processed
    """

    def __init__(self, partition_folder_path: str, extension = '.partition'):
        self.partition_folder_path = partition_folder_path
        self.extension = extension
        self.partitions: List[DatasetPartition] = []
        self.partition_file_names: List[str] = []
        self.partition_error_file_names: List[str] = []

        self.verify_partition_folder_path()
        self.load_partitions()

    def verify_partition_folder_path(self):
        """
        Verifies that the attribute `self.partition_folder_path` is correctly pointing
        to a folder in the file system
        """
        if not os.path.exists(self.partition_folder_path):
            raise FileNotFoundError(f'This is not a valid path : {self.partition_folder_path}')
        
        if not os.path.isdir(self.partition_folder_path):
            raise NotADirectoryError(f'The path is not a directory : {self.partition_folder_path}')

        if self.partition_folder_path[-1] != '/':
            self.partition_folder_path += '/'


    def load_partitions(self):
        """
        Loads the partitions that are available in `self.partition_folder_path`. This will assume
        that all files ending with `self.extension` are `DatasetPartition` objects serialized with
        joblib
        """
        partition_file_names = [f for f in os.listdir(self.partition_folder_path) if os.path.isfile(os.path.join(self.partition_folder_path, f))]
    
        for partition_file_name in partition_file_names:
            if len(partition_file_name) < len(self.extension) or partition_file_name[-len(self.extension):] != self.extension:
                continue

            logger.info(f'Loading {partition_file_name}')
            try:
                partition = DatasetPartition.from_save(
                    path=os.path.join(self.partition_folder_path, partition_file_name),
                    load_original_dataset=False
                )
                self.partitions.append((partition_file_name, partition))
            except EOFError:
                self.partition_error_file_names.append(partition_file_name)
                logger.warning(f'Partition could not be loaded {partition_file_name}')

        self.partitions.sort(key=lambda x: x[1].start)
        self.partition_file_names = list(map(lambda x: x[0], self.partitions))
        self.partitions = list(map(lambda x: x[1], self.partitions))


    def get_next_partitions(self, max_partitions, full_path: bool = True):
        """
        Returns at most `max_partitions` partition file names linked to partitions that
        are not done being processed.

        Args:
            max_partitions: Number of jobs that need a partition to process
            full_path: Whether to return the full path of the partitions or just the file name

        Returns:
        List of strings containing the partitions that are not processed
        """
        results = []
        for partition_file_name, partition in zip(self.partition_file_names, self.partitions):
            if not partition.is_completed():
                results.append(partition_file_name)

            if len(results) >= max_partitions:
                break

        if full_path:
            return list(map(lambda x: self.partition_folder_path + x, results))

        return results

    def show_partitions(self, reduce_factor: int = 5):
        """
        Prints an overview of all the partitions loaded showing the percentage of completion of each

        Args:
            reduce_factor: Number of elements processed per '=' in the bar
        """
        print('Partitions')
        print('=' * 100)

        if len(self.partitions) == 0:
            print('No partitions found')
            print('=' * 100)
            return
        
        for partition in self.partitions:
            unprocessed_length = len(partition.get_unprocessed_ids())
            processed_length = partition.nb_elements - unprocessed_length
            percentage_processed = processed_length / partition.nb_elements * 100
            processed_bar_string = '=' * (processed_length // reduce_factor) + ' ' * (unprocessed_length // reduce_factor)
            color = Color.GREEN if percentage_processed > 50 else Color.RED
            print(f'{partition.start}-{partition.end}: {color}[{processed_bar_string}]{Color.OFF} {percentage_processed:.1f}% ({processed_length} / {partition.nb_elements})')

        print('=' * 100)

    def fix_broken_partitions(self):
        """
        Fixes all broken partitions that were corrupted by regenerating the partitions that were
        corrupted and keeping those that were not corrupted (with their results). This only works
        if at least one partition was loaded correctly. Otherwise, simply regenerate all partititions
        using a `Dataset` object with the function `.partition()`
        """

        for partition_file_name in self.partition_error_file_names:
            logger.info(f'Fixing {partition_file_name}')
            os.remove(self.partition_folder_path + partition_file_name)

        dataset = Dataset(self.partitions[0].original_dataset_path)
        dataset.partition(
            output_folder_path=self.partition_folder_path,
            size_of_partition=self.partitions[0].nb_elements
        )

class MimicDataset(Dataset):
    CLINICAL_NOTE_COLUMN = 'TEXT'
    DOMAIN_COLUMN = 'CATEGORY'
    CLINICAL_NOTE_ID_COLUMN = 'ROW_ID'

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.verify()

    def verify(self):
        """Verifies that the data loaded is conform to the extraction dataset template"""
        error = f'The dataset is not a valid Mimic dataset, missing the column : '
        assert self.CLINICAL_NOTE_COLUMN in self.data.columns, error + self.CLINICAL_NOTE_COLUMN
        assert self.CLINICAL_NOTE_ID_COLUMN in self.data.columns, error + self.CLINICAL_NOTE_ID_COLUMN
        assert self.DOMAIN_COLUMN in self.data.columns, error + MimicDataset.DOMAIN_COLUMN

    def sample(self, text_only: bool = True, domain: str = None):
        """
        Samples a random clinical note in the dataset

        Args:
            text_only: Whether to return the clinical note only or return the whole row
            domain: From which domain to sample clinical notes from. If None, the domain is not taken into account in the sampling

        Returns:
        Tuple containing (index of row in the dataframe, clinical note or row associated to clinical note)
        """
        df = self.data
        
        if domain is not None:
            domains = self.data[self.DOMAIN_COLUMN].unique().tolist()
            assert domain in domains, f'The domain "{domain}" is not a valid domain. Here are the valid domains : {domains}'
            df = df[df[self.DOMAIN_COLUMN] == domain]

        sample_index = random.randrange(len(df))
        row = df.iloc[sample_index]

        return (sample_index, row[MimicDataset.CLINICAL_NOTE_COLUMN]) if text_only else (sample_index, row)

    def __getitem__(self, param: Tuple[int, bool]):
        index, text_only = param
        assert index < len(self.data)
        return self.data[self.CLINICAL_NOTE_COLUMN].iloc[index] if text_only else self.data.iloc[index]

class ExtractionDataset(Dataset):
    """
    Dataset format used to store the extractions of each clinical note.

    The dataset should have at least the column 'TEXT' and one of the following columns : 'normal', 'beam' or 'constrained'.

    The column `column` corresponds to the column that contains the extractions. An extraction dataset column always contains dictionaries
    """
    CLINICAL_NOTE_COLUMN = 'TEXT'
    CLINICAL_NOTE_ID_COLUMN = 'ROW_ID'

    def __init__(self, column: str, dataset_path: str = None, data: pd.DataFrame = None):
        super().__init__(dataset_path=dataset_path, data=data)

        self.column = column
        self.verify()
        self.prepare()

    def verify(self):
        """Verifies that the data loaded is conform to the extraction dataset template"""
        error = f'The dataset is not a valid extraction dataset, missing the column : '
        assert 'TEXT' in self.data.columns, error + 'TEXT'
        assert 'normal' in self.data.columns or 'beam' in self.data.columns or 'constrained' in self.data.columns, error + 'normal, beam or constrained'
        assert self.column in ['normal', 'beam', 'constrained'], f'The column "{self.column}" is not a valid column. Here are the valid columns : normal, beam, constrained'

    def result_columns(self,):
        return [self.column]

    def prepare(self):

        for column in self.result_columns():
            self.data[column] = self._get_extractions(column)

    def _get_extractions(self, column):
        """
        Returns the extractions of a column converted to a dictionary
        Args:
            column: Column containing the extractions
        """
        extractions = []
        for extract in self.data[column]:
            if isinstance(extract, dict):
                extractions.append(extract)
                continue
            try:
                extraction = ast.literal_eval(extract)
                if isinstance(extraction, list):
                    extraction = extraction[0]
            except Exception as e:
                extraction = {}
            extractions.append(extraction)
        return extractions


    def clinical_notes(self):
        return self.data['TEXT'].tolist()

    def clinical_note_column(self):
        return self.CLINICAL_NOTE_COLUMN

    def clinical_note_id_column(self):
        return self.CLINICAL_NOTE_ID_COLUMN

class PrunedConceptDataset(ExtractionDataset):
    """
    Dataset format used to store the pruned concepts of each clinical note.

    The column `column` corresponds to the column that contains the pruned extractions.
    """
    def __init__(self, columns: List[str], dataset_path: str = None, data: pd.DataFrame = None):
        self.columns = columns
        super().__init__(column=columns[0], dataset_path=dataset_path, data=data)

    def verify(self):
        for column in self.columns:
            assert PrunedConceptDataset.valid_pruned_concept_column(column), f'The column "{column}" is not a valid column. It should be of a valid PrunedConceptDataset input column'

    def result_columns(self):
        return self.columns

    @staticmethod
    def valid_pruned_concept_column(column: str):
        elements = column.split('_')
        return len(elements) >= 2 and elements[0] in ['normal', 'beam', 'constrained']

class VerbalizedExtractionDataset(Dataset):
    """
    Dataset format used to store the verbalized extractions of each clinical note.

    The column `column` corresponds to the column that contains the verbalized extractions. It should have the following format :
    [decoding_strategy]_[domain]_verbalized

    Optionally, the inference input columns can be provided in the dataset (needed to filter non valid generations). The inference
    input columns should have the following format : [decoding_strategy]_[domain]_verbalizer_prompt
    """
    def __init__(self, columns: List[str], dataset_path: str = None, data: pd.DataFrame = None):
        super().__init__(dataset_path=dataset_path, data=data)
        self.columns = columns
        self.verify()
        self.infer()

    def infer(self):
        """
        Infers the input columns, the inference input columns, the domains and the methodology

        The input columns are the pruned columns with the _verbalizer_prompt suffix.
        The inference input columns are the input columns with the _verbalizer_prompt suffix.
        The domains are the middle elements of the column names.
        The methodologies are the first elements of the column names.
        """
        self.pruned_columns = list(map(lambda x: '_'.join(x.split('_')[:-1]), self.columns))
        self.inference_input_columns = list(map(lambda x: x + '_verbalizer_prompt', self.pruned_columns)) 
        self.domains = list(map(lambda x: '_'.join(x.split('_')[1:-1]), self.columns))
        self.methodologies = list(map(lambda x: x.split('_')[0], self.columns))

    def verify(self):
        for column in self.columns:
            assert VerbalizedExtractionDataset.valid_verbalized_column(column), f'The column "{column}" is not a valid column. It should be of a valid VerbalizedExtractionDataset input column'

    def result_columns(self):
        return self.columns

    def filter_non_valid_generations(self):
        """
        Filters the non valid generations in the dataset. A valid generation is a generation in the output column that is not NaN in the inference input column.
        Having N/A in the inference input column means that there was no important information in the clinical note to generate a summary for a domain.

        Args:
            input_columns: The input columns to filter the non valid generations from
            output_columns: The output columns to filter the non valid generations from
        """
        for inference_input_column in self.inference_input_columns:
            assert inference_input_column in self.data.columns, f'The column "{inference_input_column}" is not present in the dataset. This column which corresponds\
                to the input of the inference pipeline is needed to filter non valid generations'

        for input_column, output_column in zip(self.inference_input_columns, self.columns):
            mask = self.data[input_column].isna()
            self.data.loc[mask, output_column] = None
        return self.data

    @staticmethod
    def valid_verbalized_column(column: str):
        elements = column.split('_')
        return len(elements) >= 2 and elements[0] in ['normal', 'beam', 'constrained'] and elements[-1] == 'verbalized'

class ComparisonExtractionDataset(ExtractionDataset):
    """
    Dataset format used to store the extractions of each clinical note.

    The dataset should have at least these columns : 'TEXT', 'normal', 'beam' and 'constrained'
    """

    RESULT_COLUMNS = ['normal', 'beam', 'constrained']

    def __init__(self, dataset_path: str, data: pd.DataFrame = None):
        super().__init__(column='normal', dataset_path=dataset_path, data=data)

        self.verify()
        self.prepare()
        self.column = 'normal'

    def verify(self):
        """Verifies that the data loaded is conform to the extraction dataset template"""
        error = f'The dataset is not a valid comparison extraction dataset, missing the column : '
        assert 'TEXT' in self.data.columns, error + 'TEXT'
        assert 'normal' in self.data.columns, error + 'normal'
        assert 'beam' in self.data.columns, error + 'beam'
        assert 'constrained' in self.data.columns, error + 'constrained'

    def result_columns(self):
        return ComparisonExtractionDataset.RESULT_COLUMNS

    def greedy_extractions(self):
        return self.data[self.RESULT_COLUMNS[0]]
    
    def beam_extractions(self):
        return self.data[self.RESULT_COLUMNS[1]]
    
    def constrained_extractions(self):
        return self.data[self.RESULT_COLUMNS[2]]
