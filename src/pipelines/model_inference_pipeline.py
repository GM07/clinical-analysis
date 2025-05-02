from typing import List
import logging

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

from src.models.utils import load_model, load_tokenizer
from src.utils import batch_elements, run_inference

def apply_chat_template(tokenizer, inputs):
    """
    Applies the chat template to the inputs
    """
    return tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

class HFModelInferencePipeline:
    """
    Helper class to load HuggingFace models and perform inference using transformers
    """
    def __init__(self, model_path: str, tokenizer_path: str = None):

        if tokenizer_path is None:
            tokenizer_path = model_path

        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)

    def run_inference(self, inputs: List[str], max_new_tokens: int = 128, batch_size: int = 1):
        """
        Runs inference of a model on a set of inputs

        Args:
            inputs: Inputs to run inference on
            batch_size: Number of inputs to run inference on at once
            max_new_tokens: Number of tokens to generated
        """

        batched_prompts = batch_elements(inputs, batch_size)
        results = []
        for batch in tqdm(batched_prompts, desc='Running inference', total=len(batched_prompts)):
            encodeds = self.tokenizer(batch, return_tensors="pt", padding=True)
            model_inputs = encodeds.to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
            decoded = self.tokenizer.batch_decode(generated_ids)
            results.extend(decoded)

        return results

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)

class ModelInferencePipeline:

    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.nb_gpus = torch.cuda.device_count()
        logger.info(f'Using {self.nb_gpus} GPUs')

        if tokenizer_path is None:
            tokenizer_path = model_path

        self.llm = LLM(model=model_path, tokenizer=tokenizer_path, tensor_parallel_size=self.nb_gpus)
        self.tokenizer = load_tokenizer(tokenizer_path)

    def run_inference(self, inputs: List, max_new_tokens: int = 128):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
        """
        params = SamplingParams(max_tokens=max_new_tokens)

        outputs = self.llm.generate(inputs, sampling_params=params)

        return [output.outputs[0].text for output in outputs]

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)

class ClassifierModelInferencePipeline:

    def __init__(self, model_path: str, tokenizer_path: str = None):
        if tokenizer_path is None:
            self.model = LLM(model=model_path, tokenizer=model_path, task='classify', max_seq_len_to_capture=8192)
        else:
            self.model = LLM(model=model_path, tokenizer=tokenizer_path, task='classify', max_seq_len_to_capture=8192)

    def run_inference(self, inputs: List, batch_size: int = 16, apply_chat_template: bool = False):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            batch_size: Number of inputs to process in parallel
        """
        predictions = []

        if apply_chat_template:
            inputs = self.apply_chat_template(inputs)

        batches = batch_elements(inputs, batch_size=batch_size)
        for batch in tqdm(batches, total=len(batches), desc='Processing batches'):
            outputs = self.model.classify(batch)
            predictions.extend([output.outputs.probs for output in outputs])

        return predictions

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)
