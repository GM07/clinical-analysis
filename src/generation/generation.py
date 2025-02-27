from typing import Dict, List
from collections import Counter, defaultdict
import logging
import types

from tqdm import tqdm

import torch
from accelerate import Accelerator
from transformers.generation.configuration_utils import GenerationConfig as HFGenerationConfig
from transformers.generation.utils import GenerationMixin

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import OntologyBeamScorer, OntologyBeamScorerConfig, GenerationInput, GenerationConfig
from src.generation.chat_template import ChatTemplate
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed
from src.generation.templates import BASE_PROMPT_TEMPLATE
from src.generation.custom_generation import custom_generate

logger = logging.getLogger(__name__)

class OntologyConstrainedModel:

    def __init__(
        self, 
        model, 
        tokenizer, 
        snomed: Snomed, 
        annotator: Annotator, 
        accelerator: Accelerator = None,
        apply_chat_template: bool = True
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer

        self.chat_template = ChatTemplate(tokenizer)
        self.snomed = snomed
        self.annotator = annotator
        self.accelerator = accelerator
        self.apply_chat_template = apply_chat_template

        self.model.eval()

    def get_device(self):
        """
        Returns the accurate device based on whether an accelerator object was provided in the constructor. If it was
        provided, it will return the accelerator object's device. If not, it will return the model's device
        """
        return self.model.device if self.accelerator is None else self.accelerator.device

    def normal_generate(self):
        """
        Sets the `generate` method of the model to the default one
        """
        self.model.generate = types.MethodType(GenerationMixin.generate, self.model)

    def modified_generate(self):
        """
        Sets the `generate` method of the model to the modified one (allowing to input a beam search scorer)
        """
        self.model.generate = types.MethodType(custom_generate, self.model)


    def prepare_model_inputs(self, prompts: List[str]):
        """
        Prepares a list of prompts to be sent to the model by applying the chat template and tokenizing the input
        
        Args:
            prompts: List of prompts to send to the model
        """
        if self.apply_chat_template and self.tokenizer.chat_template is not None:
            prompts = self.chat_template.batched_single_user_entry(prompts)
        
        model_input = self.tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=False, 
            pad_to_multiple_of=8,
            add_special_tokens=False
        )
        return model_input

    def get_final_generation(self, prompts_input_ids: torch.Tensor, generated_answer) -> List[str]:
        """
        Formats the generations of a model by only returning the newly generated tokens and decoding them with the tokenizer

        Args:
            prompt_input_ids: List of inputs that were used by the model to get the generations
            generated_answer: Generations of the model

        Returns:
        Decoded generations without the input tokens
        """
        new_tokens = generated_answer[:, prompts_input_ids.shape[-1]:]
        results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return results


    def greedy_search(self, generation_input: GenerationInput) -> List[str]:
        """
        Sends the `generation_input` to the model using greedy search decoding

        Args:
            generation_input: Object containing the prompts to send to the model
        """

        if len(generation_input.prompts[0]) == 0:
            logger.warning(f'The prompts sent to the model are empty')
        
        model_input = self.prepare_model_inputs(generation_input.prompts)
        model_input['input_ids'] = model_input['input_ids'].to(self.get_device())
        model_input['attention_mask'] = model_input['attention_mask'].to(self.get_device())

        self.normal_generate()

        hf_generation_config = HFGenerationConfig(
                temperature=0,
                top_p=1,
                top_k=-1,
                seed=42
        )

        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=128,
                generation_config=hf_generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            final_answers = self.get_final_generation(model_input['input_ids'], generated)
            del model_input
            return final_answers
    

    def group_beam_search(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):

        tokenized_inputs = self.prepare_model_inputs(generation_input.prompts)
        input_ids = tokenized_inputs['input_ids'].to(self.get_device())
        attention_mask = tokenized_inputs['attention_mask'].to(self.get_device())
        hf_generation_config = HFGenerationConfig(
                temperature=0,
                top_p=1,
                top_k=-1,
                num_beams=generation_config.nb_beams,  # Number of beams for beam search
                num_return_sequences=1,  # Return all beams
                num_beam_groups=generation_config.nb_beam_groups,
                diversity_penalty=generation_config.diversity_penalty,
                seed=42
        )
        # generate_params = {
        #         "input_ids": input_ids,
        #         "generation_config": hf_generation_config,
        #         "return_dict_in_generate": True,
        #         "output_scores": True,
        #         "max_new_tokens": 128,
        # }

        with torch.no_grad():
            if generation_config.normal_beam_search:
                self.normal_generate()
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_generation_config,
                    max_new_tokens=generation_config.max_new_tokens
                )
            else:
                self.modified_generate()
                ontology_beam_scorer = OntologyBeamScorer(
                    config=OntologyBeamScorerConfig(
                        tokenizer=self.tokenizer,
                        annotator=self.annotator,
                        snomed=self.snomed,
                        generation_input=generation_input,
                        generation_config=generation_config
                    ),
                    batch_size=len(generation_input.prompts),
                    device=self.get_device()
                )
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_generation_config,
                    max_new_tokens=generation_config.max_new_tokens,
                    beam_scorer=ontology_beam_scorer
                )
            generated_tokens = generation_output
            final_answers = self.get_final_generation(tokenized_inputs['input_ids'], generated_tokens)
            return final_answers

    def generate(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):

        if generation_config.use_group_beam_search:
            return self.group_beam_search(generation_input, generation_config)
        else:
            return self.greedy_search(generation_input)




class OntologyPromptTemplate:

    def __init__(self, question_template: str = None):
        if question_template is None:
            self.question_template = BASE_PROMPT_TEMPLATE
        else:
            self.question_template = question_template

class OntologyBasedPrompter:
    """
    Responsable of the extraction step
    """
    
    def __init__(
        self, 
        snomed: Snomed, 
        annotator: Annotator, 
        constrained_model: OntologyConstrainedModel = None, 
        template: OntologyPromptTemplate = OntologyPromptTemplate(),
        dataset_mode: bool = False
    ):
        """
        Initializes an OntologyBasedPrompter object that handles the extraction of medical concepts from text.

        Args:
            constrained_model: An OntologyConstrainedModel instance used to generate responses
            snomed: A Snomed ontology instance providing access to medical concepts and relationships
            annotator: An Annotator instance for identifying medical concepts in text
            template: An OntologyPromptTemplate instance defining the prompt format (default: OntologyPromptTemplate())
            dataset_mode: Whether the prompt will be stored in a dataset instead of being sent to the model
        """
        
        self.constrained_model = constrained_model
        self.snomed = snomed
        self.annotator = annotator
        self.template = template
        self.dataset_mode = dataset_mode

        
        self.attributes_by_id = []

        self.exclude_ids = set(['362981000', '444677008', '419891008', '276339004', '106237007'])
        self.current_note_id = 0
    
    def start_multiple(self, clinical_notes: List[str], top_n=5, batch_size=1, generation_config: GenerationConfig = GenerationConfig(), ids: List[str] = None):
        """
        Prompts a model on multiple clinical notes

        Args:
            clinical_notes: Clinical notes used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note (not used if `self.dataset_mode` is `True`)
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
        generation_input = GenerationInput(prompts=prompts, clinical_notes=[clinical_note] * len(prompts), concept_ids=concept_ids)
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

            # label = self.snomed.get_label_from_id(concept_id)

            if len(answer.strip()) > 0 and valid_answer:
                # self.attributes[self.current_note_id][label] = answer
                self.attributes_by_id[self.current_note_id][concept_id] = answer
            else:
                # self.attributes[self.current_note_id][label] = 'N/A'
                self.attributes_by_id[self.current_note_id][concept_id] = 'N/A'
