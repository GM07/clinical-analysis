


from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.data.dataset import DatasetPartition
from datasets import Dataset as HuggingFaceDataset
import logging
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    Abstract class for inference pipelines
    """

    def __init__(self, model_path: str):
        self.llm = LLM(model=model_path, tokenizer=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

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

class HuggingFaceDatasetInferencePipeline(InferencePipeline):
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(
        self, 
        model_path: str, 
        input_column: str = 'input', 
        output_column: str = 'output',
    ):
        super().__init__(model_path)
        self.input_column = input_column
        self.output_column = output_column

    def __call__(self, dataset: HuggingFaceDataset, max_new_tokens: int = 128, apply_chat_template: bool = True):
        """
        Executes the pipeline on the partition

        Args:
            partition: Partition onto which the pipeline will be ran
            batch_size: Batch size during inference
        """
        results = []

        if apply_chat_template:
            def apply_chat_template_for_row(data):
                inputs = data[self.input_column]

                if isinstance(inputs[0], str):
                    inputs = list(map(lambda x: [{'role': 'user', 'content': x}], inputs))

                return {f'{self.input_column}_template': self.apply_chat_template(inputs)}
            dataset = dataset.map(apply_chat_template_for_row, batched=True)
            inputs = dataset[self.input_column + '_template']
        else:
            inputs = dataset[self.input_column]

        output = self.run_inference(inputs, max_new_tokens=max_new_tokens)
        results.extend(output)

        dataset = dataset.add_column(self.output_column, results)
        dataset = dataset.remove_columns(f'{self.input_column}_template')

        return dataset

class DatasetPartitionInferencePipeline(InferencePipeline):
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(
        self, 
        model_path: str, 
        input_column: str = 'input', 
    ):
        super().__init__(model_path)
        self.input_column = input_column

    def __call__(self, partition: DatasetPartition, max_new_tokens: int = 128):
        """
        Executes the pipeline on the partition. Assuming that the input_column already contains a chat template adapted to the model.

        Args:
            partition: Partition onto which the pipeline will be ran
            max_new_tokens: Maximum number of new tokens to generate
        """

        results, ids, inputs = [], [], []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):

            input = value[self.input_column]
            inputs.append(input)
            ids.append(i)

        result = self.run_inference(inputs, max_new_tokens=max_new_tokens)
        results = [(id_input, r) for id_input, r in zip(ids, result)]
        partition.save_results(results)
