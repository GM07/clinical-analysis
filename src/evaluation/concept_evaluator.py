import ast

from src.data.dataset import ExtractionDataset
from src.ontology.snomed import Snomed


class ConceptEvaluatorPromptGenerator:

    def __init__(self, comparison_dataset_path: str, column: str, snomed_path: str, snomed_cache_path: str):
        """
        Args:
            comparison_dataset_path: Path to comparison dataset
            column: Name of the column containing the extractions
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache path
        """
        self.dataset_path = comparison_dataset_path
        self.column = column
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path

        self.load()

    def load(self):
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path)
        self.dataset = ExtractionDataset(column=self.column, dataset_path=self.dataset_path)

    def generate(self):
        pass
