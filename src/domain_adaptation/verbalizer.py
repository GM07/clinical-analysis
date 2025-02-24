
from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import PrunedConceptDataset
from src.domain_adaptation.prompt_generator import PrunedConceptPromptGenerator
from src.ontology.snomed import Snomed
from src.pipelines.dataset_inference_pipeline import HuggingFaceDatasetInferencePipeline


class Verbalizer:

    def __init__(self, model_path: str, input_columns: list[str], snomed: Snomed):
        self.input_columns = input_columns
        self.snomed = snomed
        self.pipeline = HuggingFaceDatasetInferencePipeline(model_path=model_path)

    def verbalize_dataset(self, dataset: PrunedConceptDataset):
        self.prompt_generator = PrunedConceptPromptGenerator(mimic=dataset.data, snomed=self.snomed, input_columns=self.input_columns)
        data = self.prompt_generator.generate_prompts()
        hf_dataset = HuggingFaceDataset.from_pandas(data)
        for input_column in self.input_columns:
            hf_dataset = self.pipeline(hf_dataset, max_new_tokens=512, input_column=input_column, output_column=f'{input_column}_verbalized')
        return hf_dataset
