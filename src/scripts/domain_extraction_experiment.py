import os
import yaml

class DomainExtractionExperiment:

    REQUIRED_SECTIONS = ['dcfs', 'checkpoint', 'tokenizer', 'dataset', 'snomed', 'snomed_cache', 'medcat']

    def __init__(self, path: str) -> None:

        self.path = os.path.expandvars(path)
        self.config_path = os.path.join(self.path, 'config.yaml')

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

        self.checkpoint = self.yaml_config['checkpoint']
        self.tokenizer = self.yaml_config['tokenizer']
        self.dataset = self.yaml_config['dataset']
        self.snomed = self.yaml_config['snomed']
        self.snomed_cache = self.yaml_config['snomed_cache']
        self.medcat = self.yaml_config['medcat']
        self.batch_size = self.yaml_config.get('batch_size', 1)
        self.medcat_device = self.yaml_config.get('medcat_device', 'cuda')
        self.system_prompt = self.yaml_config.get('system_prompt', None)
        self.dataset_input_column = self.yaml_config.get('dataset_input_column', 'TEXT')

    def infer(self):

        self.internal_dataset_saving_path = os.path.join(self.path, 'internal_extractions.csv')
        self.final_results_path = os.path.join(self.path, 'results.csv')
