from typing import List
from collections import defaultdict
import logging

from tqdm import tqdm
from datasets import Dataset

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import GenerationConfig
from src.generation.ontology_prompter import OntologyPrompter

logger = logging.getLogger(__name__)

class GuidedOntologyPrompter(OntologyPrompter):
    """
    Class responsible for extracting data from clinical notes. The GuidedOntologyPrompter will prompt the model for 
    the most frequent concepts detected by the annotator in the clinical note. When prompted, the model can 
    be guided through an ontology-constrained decoding process.
    """

    def __call__(self, clinical_notes: List[str], top_n: int = 5, generation_config: GenerationConfig = GenerationConfig(), return_dataset: bool = False):
        """
        Starts the extraction on a dataset

        Args:
            clinical_notes: List of clinical notes to process
            top_n: Number of concepts to keep per note
            generation_config: Generation config to be used by the model
            return_dataset: Whether to return the internal dataset used for generation

        Returns:
        List of prompts per clinical notes
        """        
        logger.info(f'Generating prompts')
        dataset: Dataset = self.generate_dataset(clinical_notes=clinical_notes, top_n=top_n)

        return self.process_dataset(dataset, generation_config, return_dataset)

    def generate_dataset(self, clinical_notes: List[str], top_n: int) -> Dataset:
        """
        Generates the prompts needed for extraction in a dataset object
        
        Args:
            clinical_notes: List of clinical notes to process
            top_n: Number of concepts to keep per note
        """
        dataset = defaultdict(list)

        id = 0
        note_id = 0
        for clinical_note in tqdm(clinical_notes, total=len(clinical_notes), desc='Preparing dataset for processing'):

            most_frequent_concepts, _ = DomainClassFrequency.get_most_frequent_concepts(
                text=clinical_note, 
                snomed=self.snomed, 
                annotator=self.annotator, 
                top_n=top_n
            )
            
            prompts = self.create_prompts(clinical_note, most_frequent_concepts)

            for prompt, concept in zip(prompts, most_frequent_concepts):
                dataset['id'].append(id)
                dataset['note_id'].append(note_id)
                dataset['clinical_note'].append(clinical_note)
                dataset['prompt'].append(prompt)
                dataset['concept_id'].append(concept)
                id += 1

            note_id += 1
        
        return Dataset.from_dict(dataset)
