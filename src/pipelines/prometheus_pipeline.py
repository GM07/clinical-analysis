

from tqdm import tqdm
from src.dataset import DatasetPartition
from src.model_registry import LoadingConfig, ModelRegistry
from src.utils import run_inference


class PrometheusEvaluationPipeline:
    """
    Evaluates extractions on clinical notes from the mimic dataset using the Prometheus model
    """
    def __init__(self, checkpoint_path: str, loading_config: LoadingConfig = LoadingConfig()):
        """
        Args:
            checkpoint_path: Path to the HuggingFace checkpoint of the model Prometheus
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
        results = []
        ids = []
        prompts = []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            prompt = value['prompt']
            prompts.append(prompt)
            ids.append(i)

            if i % batch_size == 0 and i != 0:
                result = run_inference(self.model, self.tokenizer, prompts, max_new_tokens=256)
                results = [(id_prompt, r) for id_prompt, r in zip(ids, result)]
                partition.save_results(results)
                prompts = []
                ids = []
        results = [(id_prompt, r) for id_prompt, r in zip(ids, result)]
        partition.save_results(results)
