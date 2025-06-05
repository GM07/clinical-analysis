import os
from itertools import groupby
from operator import itemgetter
from typing import List

import logging
from datasets import Dataset
import joblib
from tqdm import tqdm

from src.generation.ontology_beam_scorer import GenerationConfig, GenerationInput
from src.generation.ontology_constrained_model import OntologyConstrainedModel, OntologyPromptTemplate
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

class OntologyPrompter:
    """
    Abstract Class responsible for extracting data from clinical notes.
    """
    def __init__(
        self, 
        snomed: Snomed, 
        constrained_model: OntologyConstrainedModel, 
        annotator: Annotator = None, 
        template: OntologyPromptTemplate = OntologyPromptTemplate(),
        system_prompt: str = None,
        log_path: str = None
    ):
        """
        Initializes an GuidedOntologyPrompter object that handles the extraction of medical concepts from text.

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
            log_path: Path to a logging file where logs will be stored (if not provided, logs are not stored)            
        """
        self.constrained_model = constrained_model
        self.snomed = snomed
        self.annotator = annotator
        self.template = template
        self.system_prompt = system_prompt
        self.log_path = log_path

    def process_dataset(
        self, 
        dataset: Dataset, 
        generation_config: GenerationConfig = GenerationConfig, 
        return_dataset: bool = True,
        prompt_col: str = 'prompt',
        clinical_note_col: str = 'clinical_note',
        concept_id_col: str = 'concept_id'
    ):
        logs = []
        results = []
        batch_size = generation_config.batch_size
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset) // batch_size + 1, desc='Processing batches...'):
            generation_input = GenerationInput(
                prompts=batch[prompt_col], 
                clinical_notes=batch[clinical_note_col], 
                concept_ids=batch[concept_id_col],
                system_prompt=self.system_prompt,
            )
            
            generation_config.batch_size = len(generation_input.prompts)
            generation_config.log = self.log_path is not None
            answers = self.constrained_model.generate(generation_input, generation_config)
            answers = self.process_answers(answers)
            results.extend(answers)
            
            if self.log_path:
                logs.append(generation_config.logs)
        
        dataset = dataset.add_column('result', results)
        
        if len(logs) > 0:
            joblib.dump(logs, self.log_path)

        answers = self.group_results_by_notes(dataset)
        
        if return_dataset:
            return answers, dataset
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

    def create_prompts(self, clinical_note: str, concept_ids: List[str]):
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
                'clinical_note': clinical_note,
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
            property_sentence = '' if len(current_property_knowledge.strip()) == 0 else f'\n{concept_label} is characterized by : \n- {current_property_knowledge}\n'
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
