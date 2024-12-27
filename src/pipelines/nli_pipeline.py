

from tqdm import tqdm
from src.data.dataset import DatasetPartition
from src.model_registry import LoadingConfig, ModelRegistry
from src.utils import run_inference

class NLIDataset:
    pass


class NLIEvaluationPipeline:
    """
    Evaluates extractions on clinical notes from the mimic dataset using an NLI Model
    """
    def __init__(
        self, 
        checkpoint_path: str,
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the HuggingFace checkpoint of the NLI model
            loading_config: Configuration object on how to load the model
        """
        self.checkpoint_path = checkpoint_path
        self.loading_config = loading_config

        self.load()
    
    def load(self):
        """Loads the model and the ontology"""
        self.model, self.tokenizer = ModelRegistry.load_single_checkpoint(self.checkpoint_path, loading_config=self.loading_config)


    def __call__(self, partition: DatasetPartition, batch_size: int = 4):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            batch_size: Batch size during inference
        """
        pass
