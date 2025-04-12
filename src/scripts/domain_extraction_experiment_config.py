import os
import yaml
import logging

from src.data.dataset import Dataset, DatasetPartition, DatasetPartitionAnalyzer


logger = logging.getLogger(__name__)

class DomainExtractionExperimentConfig:

    REQUIRED_SECTIONS = ['dcfs', 'checkpoint', 'tokenizer', 'dataset', 'snomed', 'snomed_cache', 'medcat']
    PARTITION_FOLDER_GREEDY = 'partitions_greedy'
    PARTITION_FOLDER_BEAM = 'partitions_beam'
    PARTITION_FOLDER_CONSTRAINED = 'partitions_constrained'
    CONFIG_FILE = 'config.yaml'
    PARTITION_FILE_LOCK = 'lock.partition'

    def __init__(self, path: str) -> None:

        self.path = os.path.expandvars(path)
        self.config_path = os.path.join(self.path, self.CONFIG_FILE)
        self.partition_folder_path_greedy = os.path.join(self.path, self.PARTITION_FOLDER_GREEDY)
        self.partition_folder_path_beam = os.path.join(self.path, self.PARTITION_FOLDER_BEAM)
        self.partition_folder_path_constrained = os.path.join(self.path, self.PARTITION_FOLDER_CONSTRAINED)

        self.partition_folder_paths = [
            self.partition_folder_path_constrained,
            self.partition_folder_path_beam, 
            self.partition_folder_path_greedy, 
        ]

        self.verify()
        self.infer()

    def verify(self):
        assert os.path.exists(self.config_path), 'The file config.yaml must exist in an experiment folder'

        with open(self.config_path, 'r') as f:
            try:
                self.yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file {self.config_path}: {e}")
        
        for section in self.REQUIRED_SECTIONS:
            assert section in self.yaml_config, f'Section {section} must be provided'

        self.dcfs = self.yaml_config['dcfs']
        assert isinstance(self.dcfs, list), 'The dcfs must be a list'

        self.checkpoint = os.path.expandvars(self.yaml_config['checkpoint'])
        self.tokenizer = os.path.expandvars(self.yaml_config['tokenizer'])
        self.dataset = os.path.expandvars(self.yaml_config['dataset'])
        self.snomed = os.path.expandvars(self.yaml_config['snomed'])
        self.snomed_cache = os.path.expandvars(self.yaml_config['snomed_cache'])
        self.medcat = os.path.expandvars(self.yaml_config['medcat'])
        self.batch_size = self.yaml_config.get('batch_size', 1)
        self.medcat_device = self.yaml_config.get('medcat_device', 'cuda')
        self.system_prompt = self.yaml_config.get('system_prompt', None)
        self.dataset_input_column = self.yaml_config.get('dataset_input_column', 'TEXT')
        self.apply_chat_template = self.yaml_config.get('apply_chat_template', True)
        self.partition_size = self.yaml_config.get('partition_size', 20)
        self.nb_jobs = self.yaml_config.get('partition_size', 20)

    def infer(self):

        self.final_results_path = os.path.join(self.path, 'results.csv')

        partitioned = any(map(lambda x: not os.path.exists(x), self.partition_folder_paths))

        if partitioned:
            logger.info('Dataset was not partitioned, partitioning...')
            for partition_folder_path in self.partition_folder_paths:
                dataset = Dataset(self.dataset)
                dataset.partition(
                    output_folder_path=partition_folder_path, 
                    size_of_partition=int(self.partition_size)
                )


        return self.find_next_available_partition()

    def find_next_available_partition(self):
        for partition_folder_path in self.partition_folder_paths:
            analyzer = DatasetPartitionAnalyzer(partition_folder_path)
            partition_paths = analyzer.get_next_partitions()
            for partition_path in partition_paths:
                lock_file = partition_path.replace('.partition', '.lock')
                if os.path.exists(lock_file):
                    continue
                else:
                    method = self.get_file_name(partition_folder_path).replace('partitions_', '')
                    return method, partition_path
        
        return None, None

    def get_file_name(self, path):
        return os.path.basename(path)

    def lock_partition(self, partition_path: str):
        lock_file = partition_path.replace('.partition', '.lock')
        open(lock_file, 'a').close()

    def unlock_partition(self, partition_path: str):
        lock_file = partition_path.replace('.partition', '.lock')
        os.remove(lock_file)
