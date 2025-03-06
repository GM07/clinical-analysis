
from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import PrunedConceptDataset, VerbalizedExtractionDataset
from src.domain_adaptation.prompt_generator import PrunedConceptPromptGenerator
from src.ontology.snomed import Snomed
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline


class Verbalizer:

    def __init__(self, model_path: str, tokenizer_path: str, input_columns: list[str], snomed: Snomed):
        self.input_columns = input_columns
        self.snomed = snomed
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def verbalize_dataset(self, dataset: PrunedConceptDataset):
        self.prompt_generator = PrunedConceptPromptGenerator(mimic=dataset.data, snomed=self.snomed, input_columns=self.input_columns)
        data = self.prompt_generator.generate_prompts()

        # More efficient for inference
        hf_dataset = HuggingFaceDataset.from_pandas(data)
        self.pipeline = ModelDatasetInferencePipeline(model_path=self.model_path, tokenizer_path=self.tokenizer_path)
        for input_column in self.input_columns:
            hf_dataset = self.pipeline(hf_dataset, max_new_tokens=512, input_column=f'{input_column}_verbalizer_prompt', output_column=f'{input_column}_verbalized')
        return hf_dataset

    @staticmethod
    def get_verbalizer_columns_from_pruned(columns):
        """
        Infers the verbalizer columns from the pruned columns

        Args:
            columns (list[str]): The columns to infer the verbalizer columns from (pruned columns in the form [decoding_strategy]_[domain])

        Returns:
            tuple[list[str], list[str]]: The input and output columns
        """
        input_columns = list(map(lambda x: x + '_verbalizer_prompt', columns))
        output_columns = list(map(lambda x: x + '_verbalized', columns))
        return input_columns, output_columns
