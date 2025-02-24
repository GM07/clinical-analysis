
from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import PrunedConceptDataset
from src.domain_adaptation.prompt_generator import PrunedConceptPromptGenerator
from src.ontology.snomed import Snomed
from src.pipelines.dataset_inference_pipeline import HuggingFaceDatasetInferencePipeline


class Verbalizer:

    def __init__(self, model_path: str, input_column: str, snomed: Snomed):

        self.input_column = input_column

        self.snomed = snomed
        self.pipeline = HuggingFaceDatasetInferencePipeline(model_path=model_path, input_column=input_column, output_column=f'{input_column}_verbalized')

    def verbalize_dataset(self, dataset: PrunedConceptDataset) -> HuggingFaceDataset:
        self.prompt_generator = PrunedConceptPromptGenerator(mimic=dataset.data, snomed=self.snomed, input_column=self.input_column)
        data = self.prompt_generator.generate_prompts()

        hf_dataset = HuggingFaceDataset.from_pandas(data)
        hf_dataset = self.pipeline(hf_dataset)
        return hf_dataset
