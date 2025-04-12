from typing import List, Set
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import logging

from tqdm import tqdm
from datasets import Dataset

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_prompter import OntologyConstrainedModel, OntologyPromptTemplate
from src.generation.ontology_beam_scorer import GenerationInput, GenerationConfig
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

class DomainOntologyPrompter:
    # v2.0 : Supports note batching

    def __init__(
        self, 
        snomed: Snomed, 
        annotator: Annotator, 
        constrained_model: OntologyConstrainedModel, 
        template: OntologyPromptTemplate = OntologyPromptTemplate(),
        system_prompt: str = None,
        guide_with_annotator: bool = False,
    ):
        """
        Initializes an OntologyPrompter object that handles the extraction of medical concepts from text.

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
        self.guide_with_annotator = guide_with_annotator
        self.constrained_model = constrained_model
        self.snomed = snomed
        self.annotator = annotator
        self.template = template
        self.system_prompt = system_prompt
        
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
        logger.info(f'Generating prompts')
        dataset: Dataset = self.generate_dataset(clinical_notes=clinical_notes, domain_concept_ids=domain_concept_ids)

        # results = np.unique(dataset['note_id'])
        results = []
        logger.info('Generating prompts')
        batch_size = generation_config.batch_size
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset) // batch_size + 1, desc='Processing batches...'):
            generation_input = GenerationInput(
                prompts=batch['prompt'], 
                clinical_notes=batch['clinical_note'], 
                concept_ids=batch['concept_id'],
                system_prompt=self.system_prompt
            )
            
            generation_config.batch_size = len(generation_input.prompts)
            answers = self.constrained_model.generate(generation_input, generation_config)
            answers = self.process_answers(answers)
            results.extend(answers)
        
        dataset = dataset.add_column('result', results)

        answers = self.group_results_by_notes(dataset)
        
        if return_dataset:
            small_dataset = dataset.remove_columns(['prompt']) # Prompt can be reconstructed
            return answers, small_dataset
        return answers

    def process_answers(self, answers):
        final_answers = []
        for answer in answers:
            comparable_answer = answer.strip().lower()
            if 'n/a'.lower() in comparable_answer or 'no information' in comparable_answer or 'does not mention' in comparable_answer or 'no mention' in comparable_answer:
                # Some models do not follow instructions to simply answer N/A, so we need to verify more cases
                final_answers.append('N/A')
            else:
                final_answers.append(answer.strip())
        return final_answers

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
        for clinical_note in tqdm(clinical_notes, total=len(clinical_notes), desc='Preparing dataset for processing'):

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

    def create_prompts(self, clinical_note, concept_ids: List[str]):
        """
        Creates prompts using a clinical note and concept ids before sending it to the model.

        Args:
            clinical_note: Note used in the extraction step
            concept_ids: Concept ids to create prompts from
        """

        prompts = []
        for concept_id in concept_ids:
            label = self.snomed.get_label_from_id(concept_id)
            properties = self.augment_prompt_with_ontology(concept_id, label)
            
            prompt = self.template.question_template.format_map({
                'clinical_note': clinical_note.strip(),
                'label': label,
                'properties': properties
            })

            prompts.append(prompt)
        return prompts

    def augment_prompt_with_ontology(self, concept_id: str, concept_label: str):
        """
        Augments the prompt with information found from the ontology about a concept

        Args:
            concept_id: Id of the concept used to augment the prompt
            concept_label: Label of the concept used to augment the prompt

        Returns:
        Augmented prompt
        """
        restriction_properties = self.snomed.get_restriction_properties_of_id(concept_id)
        if len(restriction_properties) == 0:
            return ''
        else:
            current_property_knowledge = '\n- '.join(map(lambda x: x.get_value(), restriction_properties))
            property_sentence = '' if len(current_property_knowledge.strip()) == 0 else f'{concept_label} is characterized by : \n- {current_property_knowledge}\n'
            return property_sentence

    @staticmethod
    def group_results_by_notes(dataset: Dataset):
        # Group by note_id and create the result list
        result = []
        for _, group in groupby(dataset, key=itemgetter('note_id')):
            # Convert group to list to work with it
            group_list = list(group)
            result.append({row['concept_id']: row['result'] for row in group_list})
        
        return result
