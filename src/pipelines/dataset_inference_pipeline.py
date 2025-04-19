from abc import abstractmethod
import time
import asyncio
import requests
from typing import Callable, List
from tqdm import tqdm

from datasets import Dataset as HuggingFaceDataset
import logging

from src.data.dataset import DatasetPartition
from src.pipelines.model_inference_pipeline import HFModelInferencePipeline, ModelInferencePipeline, apply_chat_template

logger = logging.getLogger(__name__)

class DatasetInferencePipeline:
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __call__(
        self, 
        dataset: HuggingFaceDataset, 
        input_column: str = 'input', 
        output_column: str = 'output',
        max_new_tokens: int = 256, 
        max_rows_to_process: int = None,
        apply_chat_template: bool = True, 
        rows_to_chat: Callable = None,
        saving_path: str = None,
        system_prompt: str = None,
        batch_size: int = None,
    ):
        """
        Executes an inference pipeline on a dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            max_new_tokens: Maximum number of new tokens to generate
            apply_chat_template: Whether to apply the chat template to the input
            input_column: Column to use for the input (default: 'input')
            output_column: Column to use for the output (default: 'output')
            rows_to_chat: Function to convert rows to a chat. The function should take multiple rows and return a list of messages that can be sent to the LLM.
            On message is expected per row. This is useful if the chat template needs to be created from multiple columns (few-shot, etc).
            saving_path: Path to save the results to. If None, the results are not saved.
            batch_size: Batch size to use
        """
        # Initialize results list with None values
        results = [None] * len(dataset)
        assert input_column in dataset.column_names, f'The input column "{input_column}" is not in the dataset'
        start_idx = 0

        # If output column exists, find first None value to resume from
        if output_column in dataset.column_names:
            logger.info(f"Found existing column '{output_column}' in dataset")
            existing_results = dataset[output_column]
            for i, result in enumerate(existing_results):
                if result is not None:
                    results[i] = result
                else:
                    start_idx = i
                    break
        logger.info(f"Resuming from index {start_idx}")
        
        # Calculate the end index based on max_rows_to_process
        if max_rows_to_process is None:
            # Process all remaining rows
            end_idx = len(dataset)
            logger.info(f"Processing rows from index {start_idx} to end (all remaining).")
        else:
            # Process up to max_rows_to_process rows
            end_idx = min(start_idx + max_rows_to_process, len(dataset))
            logger.info(f"Processing up to {max_rows_to_process} rows, from index {start_idx} to {end_idx}.")

        dataset = dataset.select(range(start_idx, end_idx))
        logger.info(f'Currently processing {len(dataset)} rows')

        inputs = self.get_chats(
            dataset, 
            input_column=input_column, 
            tmp_column=f'{input_column}_tmp', 
            apply_chat_template=apply_chat_template, 
            rows_to_chat=rows_to_chat,
            system_prompt=system_prompt
        )

        logger.info(f"Generated chats")

        # Only process inputs starting from start_idx
        if start_idx < len(inputs):

            args = {
                'inputs': inputs,
                'max_new_tokens': max_new_tokens
            }
            if batch_size:
                args['batch_size'] = batch_size

            output = self.run_inference(**args)
            
            output_idx = 0
            for i in range(start_idx, end_idx):
                if output_idx < len(output):
                    results[i] = output[output_idx]
                    output_idx += 1
                else:
                    # This shouldn't happen if run_inference returns one output per input
                    logger.error(f"Mismatch between expected outputs and actual outputs at index {i}")
                    break # Or handle error appropriately

            # Remove old output column if it exists, then add the updated one
            if output_column in dataset.column_names:
                dataset = dataset.remove_columns(output_column)
            dataset = dataset.add_column(output_column, results)

            # Clean up temporary columns if they were added by get_chats
            if f'{input_column}_tmp' in dataset.column_names:
                dataset = dataset.remove_columns(f'{input_column}_tmp')

        if saving_path is not None:
            dataset.to_csv(saving_path, index=False)

        return dataset

    def get_chats(self, dataset: HuggingFaceDataset, input_column: str = 'input', tmp_column: str = 'input_tmp', apply_chat_template: bool = True, rows_to_chat: Callable = None, system_prompt: str = None):
        """
        Gets the chats that needs to be sent to the LLM from the dataset

        Args:
            dataset: Dataset to get the chats from
            input_column: Column to get the chats from
            tmp_column: Column to store the temporary chats
            apply_chat_template: Whether to apply the chat template to the input
            rows_to_chat: Function to convert rows to a chat. The function should take multiple rows and return a list of messages that can be sent to the LLM.
            system_prompt: System prompt to use for the inference (only used if apply_chat_template is True and rows_to_chat is None)
        """
        if rows_to_chat is not None:
            # A function was given that converts rows to a chat. Apply it to the dataset and the output of this 
            # function is added to the dataset as a new column called f'{input_column}_tmp'
            logger.info('Applying custom function to generate chat')

            def apply_function(rows):
                chats = rows_to_chat(rows)
                return {tmp_column: chats}
            dataset = dataset.map(apply_function, batched=True)
        else:
            if apply_chat_template:
                # The input column is a prompt. Convert it to a chat template
                dataset = dataset.add_column(tmp_column, self.apply_chat_template_dataset(dataset, input_column, tmp_column, system_prompt))
            else:
                # The input column already contains the chat template
                logger.info('Input column already contains conversation')
                tmp_column = input_column

        return dataset[tmp_column]

    def apply_chat_template_dataset(self, dataset: HuggingFaceDataset, column: str = None, output_column: str = 'tmp', system_prompt: str = None):
        """
        Applies the chat template to a column of the dataset. New column is added to the dataset.

        Args:
            dataset: Dataset to apply the chat template to
            column: Column to apply the chat template to
            output_column: Column to store the output of the chat template
            system_prompt: System prompt to use (won't set a system prompt if None)

        Returns:
        Newly added column with the chat template applied
        """
        def create_chat_template_row(data):
            inputs = data[column]

            if isinstance(inputs, str):
                inputs = [[{'role': 'user', 'content': x}] if system_prompt is None \
                          else [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': x}] for x in inputs]

            return {output_column: inputs}
        
        dataset = dataset.map(create_chat_template_row, desc='Creating chat template')

        # templated = []
        # for batch in tqdm(dataset.iter(batch_size=32), total=len(dataset) // 32, desc='Applying chat template'):
            # for x in batch[output_column]:
                # templated.append(self.apply_chat_template(x))
        
        # dataset = dataset.remove_columns(output_column)
        # dataset = dataset.add_column(output_column, templated)
        dataset = dataset.map(lambda x: {output_column: self.apply_chat_template(x[output_column])}, desc='Applying chat template', num_proc=1)
        return dataset[output_column]
    
    @abstractmethod
    def apply_chat_template(self, inputs):
        raise NotImplementedError('The apply_chat_template method not implemented by the subclass. If this function\
                                  does not need to be implemented, specify a rows_to_chat callable when calling the pipeline.')

class ModelDatasetInferencePipeline(DatasetInferencePipeline, ModelInferencePipeline):
    """
    Pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        ModelInferencePipeline.__init__(self, model_path, tokenizer_path)

    def apply_chat_template(self, inputs):
        return ModelInferencePipeline.apply_chat_template(self, inputs)

class HFModelDatasetInferencePipeline(DatasetInferencePipeline, HFModelInferencePipeline):
    """
    Pipeline class for LLM inference using Huggingface on a dataset's partition
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        HFModelInferencePipeline.__init__(self, model_path, tokenizer_path)

    def apply_chat_template(self, inputs):
        return HFModelInferencePipeline.apply_chat_template(self, inputs)


class ModelPartitionedInferencePipeline(ModelInferencePipeline):
    """
    General pipeline class for LLM inference using vLLM on a dataset's partition
    """

    def __init__(self, model_path: str, input_column: str = 'input', ):
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

class ProviderDatasetInferencePipeline(DatasetInferencePipeline):
    """
    Abstract class for provider-based inference pipelines (vllm, togetherAI, etc)
    """
    
    async def __call__(
        self, 
        dataset: HuggingFaceDataset, 
        input_column: str = 'input', 
        output_column: str = 'output', 
        max_new_tokens: int = 128, 
        apply_chat_template: bool = True, 
        max_rows_to_process: int = None,
        rows_to_chat: Callable = None,
        batch_size: int = 8,
        saving_path: str = None,
        system_prompt: str = None
    ):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            input_column: Column to use for the input (default: 'input')
            output_column: Column to use for the output (default: 'output')
            max_new_tokens: Maximum number of new tokens to generate
            apply_chat_template: Whether to apply the chat template to the input
            max_rows_to_process: Maximum number of rows to process
            rows_to_chat: Function to convert rows to a chat. The function should take multiple rows and return a list of messages that can be sent to the LLM.
            batch_size: Number of requests to batch together in a single API call
            saving_path: Path to save the results to. If None, the results are not saved.
            system_prompt: System prompt to use for the inference (only used if apply_chat_template is True and rows_to_chat is None)
        """
        if rows_to_chat is None:
            assert input_column in dataset.column_names, f'The input column "{input_column}" is not in the dataset'

        # Initialize results list with None values
        results = [None] * len(dataset)
        start_idx = 0

        # If output column exists, find first None value to resume from
        if output_column in dataset.column_names:
            logger.info(f"Found existing column '{output_column}' in dataset")
            existing_results = dataset[output_column]
            for i, result in enumerate(existing_results):
                if result is not None:
                    results[i] = result
                else:
                    start_idx = i
                    break
        
        print(f"Resuming from index {start_idx}")

        inputs = self.get_chats(
            dataset, input_column=input_column, 
            tmp_column=f'{input_column}_tmp', 
            apply_chat_template=apply_chat_template, 
            rows_to_chat=rows_to_chat,
            system_prompt=system_prompt
        )

        # Only process inputs starting from start_idx
        if start_idx < len(inputs):
            output = await self.run_inference(
                inputs[start_idx:], 
                max_new_tokens=max_new_tokens, 
                start_idx=start_idx, 
                max_rows_to_process=max_rows_to_process,
                batch_size=batch_size
            )
            
            # Update results list with new outputs
            for i, result in enumerate(output, start=start_idx):
                results[i] = result

            # Save intermediate results
            if output_column in dataset.column_names:
                dataset = dataset.remove_columns(output_column)
            dataset = dataset.add_column(output_column, results)

            if f'{input_column}_tmp' in dataset.column_names:
                dataset = dataset.remove_columns(f'{input_column}_tmp')

        if saving_path is not None:
            dataset.to_csv(saving_path, index=False)

        return dataset

    @abstractmethod
    def call_provider(self, request_data: dict):
        """
        Calls the provider
        """
        pass

    def prompt_to_chat(self, dataset: HuggingFaceDataset, input_column: str = 'input', tmp_column: str = 'tmp', system_prompt: str = None):
        """
        Converts a prompt to a chat format

        Args:
            dataset: Dataset to convert the prompts to chat
            input_column: Column to use for the input
            tmp_column: Column to store the output
            system_prompt: System prompt to use for the inference
        """
        def prompt_to_chat_for_row(data):
            prompts = data[input_column]
            chats = list(map(lambda x: [{'role': 'user', 'content': x}] if system_prompt is None \
                             else [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': x}], prompts))
            return {tmp_column: chats}
        dataset = dataset.map(prompt_to_chat_for_row, batched=True, desc='Converting prompts to chat conversations')
        return dataset[tmp_column]

    def apply_chat_template(self, inputs):
        return inputs

    async def run_inference(self, inputs: List, max_new_tokens: int = 128, start_idx: int = 0, max_rows_to_process: int = None, batch_size: int = 8):
        """
        Runs inference on the inputs using a vLLM endpoint with batching support
        
        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
            start_idx: Starting index for logging purposes
            max_rows_to_process: Maximum number of rows to process
            batch_size: Number of requests to batch together in a single API call
        """
        # Initialize results list
        results = []
        
        # Limit the number of rows to process if specified
        if max_rows_to_process is not None:
            inputs = inputs[:min(len(inputs), max_rows_to_process)]
        
        # Process inputs in batches
        for batch_idx in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
            batch_end = min(batch_idx + batch_size, len(inputs))
            batch = inputs[batch_idx:batch_end]
            
            success = False
            retries = 0
            MAX_RETRIES = 2
            
            while not success and retries <= MAX_RETRIES:
                try:
                    # Prepare the batch request
                    request_data = {
                        "inputs": batch,
                        "parameters": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.1,
                            "top_p": 1.0
                        }
                    }
                    
                    results.extend(await self.call_provider(request_data))
                    success = True
                    
                except Exception as e:
                    retries += 1
                    logging.warning(f"Batch request failed (attempt {retries}/{MAX_RETRIES+1}): {str(e)}")
                    if retries <= MAX_RETRIES:
                        # Wait before retry with exponential backoff
                        time.sleep(0.5 * (2 ** retries))
                    else:
                        logging.error(f"Failed to process batch after {MAX_RETRIES+1} attempts")
                        results.extend([None] * (batch_end - batch_idx))
                
                # Add a small delay between batches to avoid overwhelming the endpoint
                if success:
                    time.sleep(0.1)
            
            # Check if we need to stop early due to too many failures
            if retries > MAX_RETRIES:
                logging.error(f"Too many failures, returning partial results")
                # Fill remaining results with None
                remaining = len(inputs) - len(results)
                if remaining > 0:
                    results.extend([None] * remaining)
                break
        
        return results

class OpenAIDatasetInferencePipeline(ProviderDatasetInferencePipeline):
    """
    Pipeline class for LLM inference using OpenAI API on a dataset
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        from openai import AsyncOpenAI
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)

    async def call_provider(self, request_data: dict):
        """
        Calls the provider with batched requests for parallel processing asynchronously
        """
        
        batch = request_data['inputs']
        max_new_tokens = request_data['parameters']['max_new_tokens']
        
        async def process_message(messages):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error in OpenAI request: {str(e)}")
                return None
        
        # Process all requests concurrently
        tasks = [process_message(messages) for messages in batch]
        return await asyncio.gather(*tasks)


class TogetherAIDatasetInferencePipeline(ProviderDatasetInferencePipeline):
    """
    Pipeline class for LLM inference using Together API on a dataset
    """

    def __init__(self, model_name: str):
        from together import Together

        self.model_name = model_name
        self.client = Together()

    def call_provider(self, request_data: dict):
        """
        Calls the provider
        """
        batch = request_data['inputs']
        max_new_tokens = request_data['parameters']['max_new_tokens']
        results = []

        for messages in batch:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                stream=True
            )
        
            # Extract generated text
            generated_text = response.choices[0].message.content
            results.append(generated_text)

        return results


class VLLMEndpointDatasetInferencePipeline(ProviderDatasetInferencePipeline):
    """
    Pipeline class for LLM inference using a vLLM endpoint on a dataset
    """

    def __init__(self, endpoint_url: str, api_key: str):
        """
        Args:
            endpoint_url: URL of the vLLM endpoint
            model_name: Name of the model (optional, used for logging)
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def call_provider(self, request_data: dict):
        """
        Calls the provider
        """
        batch = request_data['inputs']

        # Make API call to vLLM endpoint
        response = requests.post(
            self.endpoint_url,
            json=request_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        )
        
        # Check if request was successful
        response.raise_for_status()
        response_data = response.json()
        
        # Extract generated texts from response
        batch_results = []
        for item in response_data["outputs"]:
            if "generated_text" in item:
                batch_results.append(item["generated_text"])
            else:
                # Fallback if the response format is different
                batch_results.append(item.get("text", None))
        
        # Ensure we have the right number of results
        if len(batch_results) != len(batch):
            raise ValueError(f"Expected {len(batch)} results, got {len(batch_results)}")
        
        return batch_results
