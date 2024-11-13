import heapq
import logging
import os
from typing import Any, List, Tuple
import joblib

from colorist import Color

from datasets import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class Dataset:
    """
    Dataset class used to load datasets, partition them between jobs and regroup partitions.
    This is only a pre-processing and post-processing class. It should not be used during inference
    """

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to dataset. For now, only csv files are supported
        """
        self.dataset_path = dataset_path
    
        self.load()

    def load(self):
        """
        Loads the dataset from the `dataset_path` attribute
        """
        self.dataset_name = os.path.basename(self.dataset_path)
        self.dataset_folder_path = self.dataset_path.replace(self.dataset_name, '')

        # self.data = load_dataset('csv', data_dir=self.dataset_folder_path, data_files={'data': self.dataset_name})['data']
        self.data = pd.read_csv(self.dataset_path)

    def partition(self, output_folder_path: str, nb_partitions: int = None, size_of_partition: int = None, max_rows: int = None):
        """
        Partitions a dataset into multiple partitions (or shards) that can be used by different jobs.

        Args:
            output_folder_path: Path where the partition
        """
        assert nb_partitions is None or size_of_partition is None, f"One of the arguments `nb_partitions` or `size_of_partition` must be null"
        assert nb_partitions is not None or size_of_partition is not None, f"One of the arguments `nb_partitions` or `size_of_partition` must be provided"

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

            partition = DatasetPartition(self.dataset_path, start, end, saving_path=output_folder_path + f'partition_{start}_{end}.partition')
            partition.save()

            current_rows_processed += end - start

            if max_rows is not None and current_rows_processed > max_rows:
                break

    @staticmethod
    def partitions_to_file(partition_folder_path: str, output_file_path: str, column_names: List[str] = ['normal', 'beam', 'constrained']):
        """
        Will create a csv file using the results of all partitions in `partition_folder_path`

        Args:
            partition_folder_path: Path of the folder containing all partitions
            output_file_path: Path where the output dataset will be saved
            column_names: Name of the columns where the results will be stored
        """
        analyzer = DatasetPartitionAnalyzer(partition_folder_path=partition_folder_path)

        if len(analyzer.partitions) == 0:
            return pd.DataFrame([])

        initial_dataset = pd.read_csv(analyzer.partitions[0].original_dataset_path)
        nb_results = len(analyzer.partitions[0].results[0])
        assert nb_results == len(column_names), 'The column names are not equal to the number of results per sample'

        for column_name in column_names:
            initial_dataset[f'{column_name}'] = None

        for partition in analyzer.partitions:
            for i, result_val in partition.results.items():
                if result_val is None:
                    continue
                for res, column_name in zip(result_val, column_names):
                    initial_dataset[column_name].iloc[partition.start + i] = res

        initial_dataset.to_csv(output_file_path, index=False)


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
        self.save()
        self.data = tmp_data

    def load(self):
        """
        Loads the dataset from the `original_dataset_path` path
        """
        self.data = pd.read_csv(self.original_dataset_path)

    def save(self, saving_path: str = None):
        """
        Saves the partition into a single file

        Args:
            saving_path: Path where the partition will be saved. This will override `self.saving_path` if provided
        """
        if saving_path is None:
            saving_path = self.saving_path

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
    def from_save(self, saving_path: str):
        """
        Loads a partition from a file

        Args:
            saving_path: Path where to load the partition from        
        """
        partition: DatasetPartition = joblib.load(saving_path)
        partition.saving_path = saving_path
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

        self.load_partitions()

    def load_partitions(self):
        """
        Loads the partitions that are available in `self.partition_folder_path`
        """
        if not os.path.exists(self.partition_folder_path):
            raise FileNotFoundError(f'This is not a valid path : {self.partition_folder_path}')
        
        if not os.path.isdir(self.partition_folder_path):
            raise NotADirectoryError(f'The path is not a directory : {self.partition_folder_path}')

        if self.partition_folder_path[-1] != '/':
            self.partition_folder_path += '/'

        partition_files = [f for f in os.listdir(self.partition_folder_path) if os.path.isfile(os.path.join(self.partition_folder_path, f))]
    
        for partition_file in partition_files:
            if len(partition_file) < len(self.extension) or partition_file[-len(self.extension):] != self.extension:
                continue
            partition = DatasetPartition.from_save(os.path.join(self.partition_folder_path, partition_file))
            self.partitions.append((partition_file, partition))

        self.partitions.sort(key=lambda x: x[1].start)
        self.partition_file_names = list(map(lambda x: x[0], self.partitions))
        self.partitions = list(map(lambda x: x[1], self.partitions))


    def get_next_partitions(self, nb_jobs, full_path: bool = True):

        """
        Returns at most `nb_jobs` partition file names that are not done being processed

        Args:
            nb_jobs: Number of jobs that need a partition to process
            full_path: Whether to return the full path of the partitions or just the file name

        Returns:
        List of strings containing the partitions that are not processed
        """
        results = []
        for partition_file_name, partition in zip(self.partition_file_names, self.partitions):
            if not partition.is_completed():
                results.append(partition_file_name)

            if len(results) >= nb_jobs:
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
