from typing import Dict, List
import logging

from datasets import Dataset
from tqdm import tqdm

from src.data.dataset import ComparisonExtractionDataset
from src.models.prometheus import Prometheus
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

class ConceptValueEvaluatorPromptGenerator:
    """
    Used to generate prompts from a set of extractions for an evaluator model to judge whether a concept and its extraction is is valid in a clinical note.

    In our case, we use Llama-70B-Instruct and Prometheus to evaluate whether the generations and concepts make sense with a clinical note.
    """

    GENERAL_TEMPLATE = """Your goal is to judge whether a model correctly extracted information on a clinical note. The model was asked to extract information related to the concept '{concept}'. If the concept is not mentioned in the clinical note, the model was asked to answer with 'N/A'.

Here is the clinical note:
{clinical_note}

Here is what the model extracted: 
{concept_extraction}

Did the model answer correctly ? Answer with Yes or No only. Do not generate anything else.
"""

    def __init__(self, dataset_path: str, snomed_path: str, snomed_cache_path: str):
        """
        Args:
            dataset_path: Path to dataset (ComparisonExtractionDataset in csv)
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache path
        """
        self.dataset_path = dataset_path
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path

        self.load()

    def load(self):
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path)
        self.dataset = ComparisonExtractionDataset(dataset_path=self.dataset_path)

    def generate(self):
        # We expect for each row a set of concepts
        normal_extractions = self.dataset.greedy_extractions()
        beam_extractions = self.dataset.beam_extractions()
        constrained_extractions = self.dataset.constrained_extractions()
        notes = self.dataset.clinical_notes()

        result = {
            'note': [],
            'concept_id': [],
            'concept_label': [],
            'extraction': [],
            'method': []
        }

        self._explode_extractions(normal_extractions, notes, result, 'greedy')
        self._explode_extractions(beam_extractions, notes, result, 'beam')
        self._explode_extractions(constrained_extractions, notes, result, 'constrained')

        dataset = Dataset.from_dict(result)
        dataset = dataset.map(lambda x: {'prompt': self.generate_prompt(x)}, desc='Generating prompts')
        dataset = dataset.remove_columns(['note'])
        return dataset

    def generate_prompt(self, x):
        label = x['concept_label']
        concept_extract = f"{x['concept_label']} : {x['extraction']}"
        clinical_note = x['note'].strip()
        return self.GENERAL_TEMPLATE.format(
            concept=label,
            clinical_note=clinical_note,
            concept_extraction=concept_extract
        )

    def _explode_extractions(self, extractions: List[Dict[str, str]], notes: List[str], result: Dict[str, List[str]], method: str):
        for method_extraction, note in tqdm(zip(extractions, notes), total=len(extractions), desc=f'Generating samples for {method} method'):
            for concept, extraction in method_extraction.items():
                label = self.snomed.get_label_from_id(concept)
                result['note'].append(note)
                result['concept_id'].append(concept)
                result['concept_label'].append(label)
                result['extraction'].append(extraction)
                result['method'].append(method)


class ConceptEvaluatorPromptGenerator(ConceptValueEvaluatorPromptGenerator):
    """
    Used to generate prompts from a set of extractions for an evaluator model to judge whether a concept and its extraction is is valid in a clinical note.

    In our case, we use Llama-70B-Instruct and Prometheus to evaluate whether the generations and concepts make sense with a clinical note.
    """

    GENERAL_TEMPLATE = """Your goal is to evaluate whether a concept is present on a clinical note.

Here is the clinical note:
{clinical_note}

Is the concept "{concept}" present in the clinical note provided ? Answer with yes or no only. Do not generate anything else.
"""
    def __init__(self, dataset_path: str, snomed_path: str, snomed_cache_path: str):
        super().__init__(dataset_path, snomed_path, snomed_cache_path)

    def generate_prompt(self, x):
        label = x['concept_label']
        label = x['concept_label']
        clinical_note = x['note'].strip()
        return self.GENERAL_TEMPLATE.format(
            concept=label,
            clinical_note=clinical_note,
        )
