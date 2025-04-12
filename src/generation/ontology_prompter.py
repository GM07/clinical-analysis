from typing import List
from collections import defaultdict
import logging

from tqdm import tqdm

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import GenerationInput, GenerationConfig
from src.generation.ontology_constrained_model import OntologyConstrainedModel, OntologyPromptTemplate
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

class OntologyPrompter:
    # v1.0 : Does not support clinical note batching, only concept batching
    
    def __init__(
        self, 
        snomed: Snomed, 
        annotator: Annotator, 
        constrained_model: OntologyConstrainedModel = None, 
        template: OntologyPromptTemplate = OntologyPromptTemplate(),
        dataset_mode: bool = False,
        system_prompt: str = None
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
        """
        
        self.constrained_model = constrained_model
        self.snomed = snomed
        self.annotator = annotator
        self.template = template
        self.dataset_mode = dataset_mode
        self.system_prompt = system_prompt
        
        self.attributes_by_id = []

        self.current_note_id = 0
    
    def start_multiple(self, clinical_notes: List[str], top_n=5, batch_size=1, generation_config: GenerationConfig = GenerationConfig(), ids: List[str] = None):
        """
        Prompts a model on multiple clinical notes

        Args:
            clinical_notes: Clinical notes used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note (not used if `self.dataset_mode` is `True`) (will override generation_config.batch_size)
            generation_config: Configuration guiding the model's generation (not used if `self.dataset_mode` is `True`)
            ids: Ids indentifying each clinical note (used if `self.dataset_mode` is `True` to store the prompts)

        Returns:
        Tuple of dictionaries where the first dictionary contains {concept_id: extraction} and the 
        second dictionary contains {concept_label: extraction}
        """

        if self.dataset_mode:
            return self.start_dataset(clinical_notes, ids, top_n=top_n)
        
        # self.attributes = []
        self.attributes_by_id = []

        for i, note in enumerate(clinical_notes):
            self.current_note_id = i
            self.attributes_by_id.append({})
            self.start(note, top_n=top_n, batch_size=batch_size, generation_config=generation_config)

            return self.attributes_by_id.copy()
    
    def start_dataset(self, clinical_notes: List[str], ids: List[str], top_n: int = 5):
        """
        Starts the extraction on a dataset

        Args:
            clinical_notes: Clinical notes used to extract information from
            ids: Ids indentifying each clinical note
            top_n: Number maximal concepts to extract from each clinical note

        Returns:
        Dictionary where the keys are the ids and the values are the prompts
        """

        assert len(ids) == len(clinical_notes), 'The number of ids should be the same as the number of clinical notes'
        
        logger.info(f'Generating prompts for dataset')
        dataset = defaultdict(list)

        for id, clinical_note in tqdm(zip(ids, clinical_notes), total=len(ids)):
            most_frequent_concepts, _ = DomainClassFrequency.get_most_frequent_concepts(
                text=clinical_note, 
                snomed=self.snomed, 
                annotator=self.annotator, 
                top_n=top_n
            )
            prompts = self.create_prompts(clinical_note, most_frequent_concepts)
            dataset[id] = prompts

        return dataset

    def start(self, clinical_note: str, top_n: int = 5, batch_size: int = 1, generation_config: GenerationConfig = GenerationConfig()):
        """
        Retrieves the most frequent concepts present in the clinical note and extracts them (or stores the prompt if `self.dataset_mode` is `True`)

        Args:
            clinical_note: Clinical note used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note (not used if `self.dataset_mode` is `True`)
            generation_config: Configuration guiding the model's generation (not used if `self.dataset_mode` is `True`)
        
        Returns:
        Dictionary containing {concept_id: extraction}
        """ 
        most_frequent_concepts, _ = DomainClassFrequency.get_most_frequent_concepts(
            text=clinical_note, 
            snomed=self.snomed, 
            annotator=self.annotator, 
            top_n=top_n
        )

        if len(most_frequent_concepts) == 0:
            return
        
        logger.debug(f'Number of concepts extracted : {len(most_frequent_concepts)}')
        logger.debug(f'Most frequent concepts : {list(map(lambda x: x.label, self.snomed.convert_ids_to_classes(most_frequent_concepts)))}')
        
        for i in range((len(most_frequent_concepts) // batch_size) + 1):
            start = i * batch_size
            end = min(len(most_frequent_concepts), (i + 1) * batch_size)
            concept_ids = most_frequent_concepts[start:end]

            self.extract_attribute(clinical_note, concept_ids, generation_config=generation_config)

    def extract_attribute(
        self, 
        clinical_note: str, 
        concept_ids: List[str], 
        generation_config: GenerationConfig = GenerationConfig()
    ):
        """
        Performs extraction step on a clinical note based on certain concepts from the ontology. It 
        then stores the extractions in `self.attributes_by_id`

        Args:
            clinical_note: Clinical note from which the information linked to concepts must be extracted
            concept_ids: Concept ids present in the clinical note guiding the extraction phase
            generation_config: Configuration guiding the model's generation
        """
        
        if len(concept_ids) == 0:
            return

        prompts = self.create_prompts(clinical_note, concept_ids)

        # Generate answers
        generation_input = GenerationInput(prompts=prompts, clinical_notes=[clinical_note] * len(prompts), concept_ids=concept_ids, system_prompt=self.system_prompt)
        generation_config.batch_size = len(prompts)
        answers = self.constrained_model.generate(generation_input, generation_config)

        self.store_extractions_from_generation(concept_ids, answers)

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
            property_sentence = '' if len(current_property_knowledge.strip()) == 0 else f'{concept_label} is characterized by : \n- {current_property_knowledge}\n'
            return property_sentence


    def store_extractions_from_generation(self, concept_ids: List[str], generations: List[str]):
        """
        Stores the extractions from the model's generations

        Args:
            concept_ids: List of concept ids used in the generation
            generations: Model's generations linked to `concept_ids`
        """

        if len(concept_ids) != len(generations):
            raise ValueError(f'Length of the questions ({len(concept_ids)}) should be the same as the length of the generations ({len(generations)})')

        for concept_id, answer in zip(concept_ids, generations):

            valid_answer = 'N/A' not in answer.strip()
            if len(answer.strip()) > 0 and valid_answer:
                self.attributes_by_id[self.current_note_id][concept_id] = answer
            else:
                self.attributes_by_id[self.current_note_id][concept_id] = 'N/A'

