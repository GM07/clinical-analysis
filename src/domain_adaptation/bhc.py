from typing import List
from datasets import Dataset

from src.generation.templates import BHC_BASE_TEMPLATE


class BHCPromptGenerator:
    """
    Class used to generate prompts for the BHC pipeline
    """

    def __init__(self, mimic_path: str):
        self.mimic_path = mimic_path

        self.dataset = Dataset.from_csv(self.mimic_path)

    def generate_prompts(self,):
        """
        Generate prompts for the BHC pipeline
        """

        def admission_to_prompt(clinical_notes: List[str]):
            prompt = BHC_BASE_TEMPLATE.format(clinical_notes=clinical_notes)
            return prompt

        self.dataset.to_pandas().groupby('HADM_ID').aggregate(admission_to_prompt)
        
