from typing import List, Optional
import logging

import torch
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

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

    def run_inference(self, inputs: List, max_new_tokens: int = 512, verify_lengths: bool = True):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
            verify_lengths: Whether to verify if prompts lengths are less than the max model length (creates errors with vLLM if that's the case, but adds overhead)
        """

        params = SamplingParams(max_tokens=max_new_tokens, temperature=0, top_k=-1)
        if not verify_lengths:

            outputs = self.llm.generate(inputs, sampling_params=params)

            return [output.outputs[0].text for output in outputs]

        # Get the maximum model length from the vLLM engine's config
        # This max_model_len is the total sequence length (prompt + generated tokens)
        # The check vLLM performs is on the prompt token IDs length itself.
        engine_model_config = self.llm.llm_engine.model_config
        vllm_max_model_len = engine_model_config.max_model_len

        results: List[Optional[str]] = [None] * len(inputs)

        valid_prompts: List[str] = []
        valid_indices: List[int] = []
        
        # Pre-filter prompts
        for i, prompt_text in enumerate(inputs):
            prompt_token_ids = self.tokenizer.encode(prompt_text)
            
            if len(prompt_token_ids) <= vllm_max_model_len - 1:
                valid_prompts.append(prompt_text)
                valid_indices.append(i)
            else:
                logger.warning(
                    f"Prompt at index {i} (token length {len(prompt_token_ids)}) "
                    f"exceeds vLLM max model length ({vllm_max_model_len}). Skipping."
                )

        if not valid_prompts:
            logger.info("No valid prompts to process after length check.")
            return results
        
        try:
            # Generate outputs only for valid prompts
            # vLLM's generate can handle a list of prompts
            generated_outputs: List[RequestOutput] = self.llm.generate(valid_prompts, sampling_params=params)
            
            # Populate results for valid prompts at their original positions
            for i, output_obj in enumerate(generated_outputs):
                original_index = valid_indices[i]
                if output_obj.outputs: # Check if there are any output sequences
                    results[original_index] = output_obj.outputs[0].text.strip()
                else:
                    logger.warning(f"No output generated for prompt at original index {original_index} (valid prompt: '{valid_prompts[i][:50]}...').")
                    results[original_index] = None # Explicitly set to None if no output
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}")
            for original_idx in valid_indices:
                results[original_idx] = None

        return results

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
