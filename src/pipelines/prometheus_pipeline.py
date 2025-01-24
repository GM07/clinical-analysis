from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
import os

from src.data.dataset import DatasetPartition
from src.model_registry import LoadingConfig, ModelRegistry
from src.utils import run_inference

import logging

logger = logging.getLogger(__name__)

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

    def run_inference(self, prompts: List[str], max_new_tokens: int = 512):
        """
        Runs inference on the prompts. 

        Args:
            prompts: List of prompts to run inference on (single batch)
            max_new_tokens: Maximum number of new tokens to generate
        """
        return run_inference(self.model, self.tokenizer, prompts, max_new_tokens=max_new_tokens)

    def __call__(self, partition: DatasetPartition, batch_size: int = 24):
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
                result = self.run_inference(prompts, max_new_tokens=256)
                results = [(id_prompt, r) for id_prompt, r in zip(ids, result)]
                partition.save_results(results)
                prompts = []
                ids = []
        results = [(id_prompt, r) for id_prompt, r in zip(ids, result)]
        partition.save_results(results)


class FastPrometheusEvaluationPipeline(PrometheusEvaluationPipeline):
    """
    Faster version of the PrometheusEvaluationPipeline that uses vllm for inference
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
        """Loads the model using vllm"""
        
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        logger.info(f"Using {gpu_count} GPUs")

        # Set multiprocessing method for vLLM workers
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # TODO : Use flash attn
        self.model = LLM(
            model=self.checkpoint_path,
            dtype='float16',
            tensor_parallel_size=gpu_count,
            # pipeline_parallel_size=1,
            # use_flash_attn=True,
            # use_safetensors=True,
        )

    def run_inference(self, prompts: List[str], max_new_tokens: int = 512):
        """
        Runs inference on the prompts using vllm

        Args:
            prompts: List of prompts to run inference on
            max_new_tokens: Maximum number of new tokens to generate
        """
        params = SamplingParams(max_tokens=max_new_tokens)
        outputs = self.model.generate(prompts, sampling_params=params)
        return [output.outputs[0].text for output in outputs]
