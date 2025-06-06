from typing import List
from collections import defaultdict
import logging

from tqdm import tqdm
from datasets import Dataset

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import GenerationConfig
from src.generation.ontology_constrained_model import OntologyConstrainedModel, OntologyPromptTemplate
from src.generation.ontology_prompter import OntologyPrompter
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

class GuidedOntologyPrompter(OntologyPrompter):
    """
    Class responsible for extracting data from clinical notes. The GuidedOntologyPrompter will prompt the model for 
    the most frequent concepts detected by the annotator in the clinical note. When prompted, the model can 
    be guided through an ontology-constrained decoding process.
    """

    def __init__(self, snomed: Snomed, constrained_model: OntologyConstrainedModel, annotator: Annotator = None, template: OntologyPromptTemplate = OntologyPromptTemplate(), system_prompt: str = None, log_path: str = None):
        super().__init__(snomed, constrained_model, annotator, template, system_prompt, log_path)

    def __call__(self, clinical_notes: List[str], top_n: int = 5, generation_config: GenerationConfig = GenerationConfig(), return_dataset: bool = False, dataset_cache: Dataset = None):
        """
        Starts the extraction on a dataset

        Args:
            clinical_notes: List of clinical notes to process
            top_n: Number of concepts to keep per note
            generation_config: Generation config to be used by the model
            return_dataset: Whether to return the internal dataset used for generation
            dataset_cache: If multiple calls are needed for this function, this argument can be used to prevent regenerating the internal dataset everytime

        Returns:
        List of prompts per clinical notes
        """  
        if dataset_cache is not None:
            logger.info(f'Using dataset cache of {len(dataset_cache)} rows')
            dataset = dataset_cache
        else:
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
        for clinical_note in tqdm(clinical_notes, total=len(clinical_notes), desc='Generating prompts'):
            
            most_frequent_concepts, _ = DomainClassFrequency.get_most_frequent_concepts(
                text=clinical_note, 
                snomed=self.snomed, 
                annotator=self.annotator, 
                top_n=top_n
            )

            if len(most_frequent_concepts) == 0:
                dataset['id'].append(id)
                dataset['note_id'].append(note_id)
                dataset['clinical_note'].append(clinical_note)
                dataset['prompt'].append('')
                dataset['concept_id'].append('')
                id += 1
            else:            
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
