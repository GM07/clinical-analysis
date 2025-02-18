


from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.data.dataset import DatasetPartition
from datasets import Dataset as HuggingFaceDataset

class HuggingFaceDatasetInferencePipeline:
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(
        self, 
        model_path: str, 
        input_column: str = 'input', 
        output_column: str = 'output',
        chat_mode: bool = True,
    ):
        self.llm = LLM(model=model_path)
        self.input_column = input_column
        self.output_column = output_column
        self.chat_mode = chat_mode

    def __call__(self, dataset: HuggingFaceDataset, batch_size: int = 24, max_new_tokens: int = 128):
        """
        Executes the pipeline on the partition

        Args:
            partition: Partition onto which the pipeline will be ran
            batch_size: Batch size during inference
        """
        results = []
        for data in dataset.batch(batch_size):
            input = data[self.input_column]
            output = self.run_inference(input, max_new_tokens=max_new_tokens)
            results.append(output)

        dataset = dataset.add_column(self.output_column, results)

        return dataset


    def run_inference(self, inputs: List, max_new_tokens: int = 128):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
        """
        params = SamplingParams(max_tokens=max_new_tokens)

        if self.chat_mode:
            outputs = self.llm.chat(inputs, sampling_params=params)
        else:
            outputs = self.llm.generate(inputs, sampling_params=params)

        return [output.outputs[0].text for output in outputs]


class DatasetPartitionInferencePipeline:
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(
        self, 
        model_path: str, 
        input_column: str = 'input', 
        chat_mode: bool = True,
    ):
        self.llm = LLM(model=model_path)
        self.input_column = input_column
        self.chat_mode = chat_mode

    def __call__(self, partition: DatasetPartition, batch_size: int = 24, max_new_tokens: int = 128):
        """
        Executes the pipeline on the partition

        Args:
            partition: Partition onto which the pipeline will be ran
            batch_size: Batch size during inference
        """

        results, ids, inputs = [], [], []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            input = value[self.input_column]
            inputs.append(input)
            ids.append(i)

            if i % batch_size == 0 and i != 0:
                result = self.run_inference(inputs, max_new_tokens=max_new_tokens)
                results = [(id_input, r) for id_input, r in zip(ids, result)]
                partition.save_results(results)
                inputs, ids = [], []

        # Save last batch        
        results = [(id_input, r) for id_input, r in zip(ids, result)]
        partition.save_results(results)

    def run_inference(self, inputs: List, max_new_tokens: int = 128):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
        """
        params = SamplingParams(max_tokens=max_new_tokens)

        if self.chat_mode:
            outputs = self.llm.chat(inputs, sampling_params=params)
        else:
            outputs = self.llm.generate(inputs, sampling_params=params)

        return [output.outputs[0].text for output in outputs]

