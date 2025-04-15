from typing import List

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import batch_elements

class ModelInferencePipeline:
    """
    Abstract class for inference pipelines
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.nb_gpus = torch.cuda.device_count()
        if tokenizer_path is None:
            self.llm = LLM(model=model_path, tensor_parallel_size=self.nb_gpus)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.llm = LLM(model=model_path, tokenizer=tokenizer_path, tensor_parallel_size=self.nb_gpus)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
        """
        Applies the chat template to the inputs
        """
        return self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

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
        """
        Applies the chat template to the inputs
        """
        return self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)


# from typing import List

# import torch
# import logging
# from tqdm import tqdm
# from vllm import LLM, SamplingParams

# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# from src.utils import batch_elements

# logger = logging.getLogger(__name__)

# class ModelInferencePipeline:
#     """
#     Abstract class for inference pipelines
#     """

#     def __init__(self, model_path: str, tokenizer_path: str = None):
#         self.nb_gpus = torch.cuda.device_count()
#         if tokenizer_path is None:
#             self.llm = LLM(model=model_path, tensor_parallel_size=self.nb_gpus)
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         else:
#             self.llm = LLM(model=model_path, tokenizer=tokenizer_path, tensor_parallel_size=self.nb_gpus)
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

#     def run_inference(self, inputs: List, max_new_tokens: int = 128):
#         """
#         Runs inference on the inputs using vllm

#         Args:
#             inputs: List of inputs to run inference on
#             max_new_tokens: Maximum number of new tokens to generate
#         """
#         params = SamplingParams(max_tokens=max_new_tokens)

#         outputs = self.llm.generate(inputs, sampling_params=params)

#         return [output.outputs[0].text for output in outputs]

#     def apply_chat_template(self, inputs: List):
#         """
#         Applies the chat template to the inputs
#         """
#         return self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

# class ClassifierModelInferencePipeline:

#     def __init__(self, model_path: str, tokenizer_path: str = None):
#         if tokenizer_path is None:
#             self.model = LLM(model=model_path, tokenizer=model_path, task='classify')
#         else:
#             self.model = LLM(model=model_path, tokenizer=tokenizer_path, task='classify')

#     def run_inference(self, inputs: List, apply_chat_template: bool = True):
#         """
#         Runs inference on the inputs using vllm

#         Args:
#             inputs: List of inputs to run inference on
#         """
#         if apply_chat_template:
#             logger.info('Applying chat template')
#             inputs = self.apply_chat_template(inputs)

#         outputs = self.model.classify(inputs)

#         return [output.outputs.probs for output in outputs]

#     def apply_chat_template(self, inputs: List):
#         """
#         Applies the chat template to the inputs
#         """
#         return self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)
