from typing import List, Set
from itertools import groupby
from operator import itemgetter
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

class DomainOntologyPrompter(OntologyPrompter):
    """
    Class responsible for extracting data from clinical notes. Given a predefined set of concepts, the DomainOntologyPrompter
    will prompt the model for every concept present in the set (or guide the prompting process with the annotator if `guide_with_annotator` = True).
    When prompted, the model can be guided through an ontology-constrained decoding process.
    """

    # v2.0 : Supports note batching

    def __init__(
        self, 
        snomed: Snomed, 
        constrained_model: OntologyConstrainedModel, 
        annotator: Annotator = None, 
        template: OntologyPromptTemplate = OntologyPromptTemplate(),
        system_prompt: str = None,
        guide_with_annotator: bool = False,
    ):
        """
        Initializes an DomainOntologyPrompter object that handles the extraction of medical concepts from text.

        Args:
            constrained_model: An OntologyConstrainedModel instance used to generate responses
            snomed: A Snomed ontology instance providing access to medical concepts and relationships
            annotator: An Annotator instance for identifying medical concepts in text
            template: An OntologyPromptTemplate instance defining the prompt format (default: OntologyPromptTemplate())
            dataset_mode: Whether the prompt will be stored in a dataset instead of being sent to the model
            system_prompt: System prompt used to generate the prompts

            guide_with_annotator: Whether to guide the prompting process with the annotator or simply prompt for every concept. 
            If True: The prompter will first tag all concepts in the clinical notes, get the ancestors and compute which concepts are present in the domain set
            If False: The prompter will ask the model to generate answers for all concepts in the domain set
        """

        OntologyPrompter.__init__(
            self, 
            snomed=snomed, 
            constrained_model=constrained_model,
            annotator=annotator,
            system_prompt=system_prompt,
            template=template,
        )
        
        self.guide_with_annotator = guide_with_annotator
        
        assert self.annotator is not None, 'An annotator must be provided if `guide_with_annotator` = True'

    def __call__(self, clinical_notes: List[str], domain_concept_ids: Set[str], generation_config: GenerationConfig = GenerationConfig(), return_dataset: bool = False):
        """
        Starts the extraction on a dataset

        Args:
            clinical_notes: List of clinical notes to process
            domain_concept_ids: Set of concept ids in snomed ontology that are related to the domain
            generation_config: Generation config to be used by the model
            return_dataset: Whether to return the internal dataset used for generation

        Returns:
        List of prompts per clinical notes
        """
        dataset: Dataset = self.generate_dataset(clinical_notes=clinical_notes, domain_concept_ids=domain_concept_ids)

        return self.process_dataset(dataset, generation_config, return_dataset)


    def generate_dataset(self, clinical_notes: List[str], domain_concept_ids: Set[str]) -> Dataset:
        """
        Generates the prompts needed for extraction in a dataset object
        
        Args:
            clinical_notes: List of clinical notes to process
            domain_concept_ids: Set of concept ids in snomed ontology that are related to the domain
        """
        dataset = defaultdict(list)

        id = 0
        note_id = 0
        for clinical_note in clinical_notes:

            domain_concepts = domain_concept_ids if not self.guide_with_annotator else DomainClassFrequency.get_domain_concepts(
                text=clinical_note, 
                snomed=self.snomed, 
                annotator=self.annotator, 
                domain_set=domain_concept_ids
            )
            
            prompts = self.create_prompts(clinical_note, domain_concepts)

            for prompt, concept in zip(prompts, domain_concepts):
                dataset['id'].append(id)
                dataset['note_id'].append(note_id)
                dataset['clinical_note'].append(clinical_note)
                dataset['prompt'].append(prompt)
                dataset['concept_id'].append(concept)
                id += 1

            note_id += 1
        return Dataset.from_dict(dataset)
