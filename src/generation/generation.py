from typing import Dict, List
from collections import Counter
import ast
import os
import logging
from dataclasses import dataclass

from colorist import Color
from tqdm import tqdm

import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria
from accelerate import Accelerator
import pandas as pd

from src.utils import clear_gpu_cache
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import OntologyBeamScorer, OntologyBeamScorerConfig, GenerationConfig
# from src.preprocessor import ClinicalNoteProcessor, preprocess_clinical_notes
from src.generation.chat_template import ChatTemplate
from src.ontology.annotator import AnnotationMatch, Annotator
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

@dataclass
class GenerationInput:

    prompts: List[str] # Clinical note + Question
    clinical_notes: List[str] # Clinical note
    concept_ids: List[str] # Concept ids

class OntologyConstrainedModel:
    """
    Model constrained by ontology during decoding process. Can't be used for inference
    normally without constrained decoding process
    """
    
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

    def get_device(self):
        """
        Returns the accurate device based on whether an accelerator object was provided in the constructor. If it was
        provided, it will return the accelerator object's device. If not, it will return the model's device
        """
        return self.model.device if self.accelerator is None else self.accelerator.device

    def verify_prompt_length(self, prompt):
        """
        Verifies that the length of a prompt is not longer than `self.max_length`

        Args:
            prompt: Prompt to verify the length of
        """
        length = len(self.tokenizer.tokenize(prompt))
        if length > self.max_length:
            logger.warning(f'Prompt longer than `self.max_length` ({self.max_length}) detected : {prompt[:100]}...')

    def prepare_model_inputs(self, prompts: List[str]):
        """
        Prepares a list of prompts to be sent to the model by applying the chat template and tokenizing the input
        
        Args:
            prompts: List of prompts to send to the model
        """
        if self.apply_chat_template:
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

    def prepare_input_for_beam_search(self, tensor: torch.Tensor, nb_beams: int, device) -> torch.Tensor:
        """
        Prepares the input for beam search by repeating and interleaving prompts `nb_beams` times.

        Args:
            tensor: List of inputs to send to the model
            nb_beams: Number of beams to be used
        """
        return torch.repeat_interleave(tensor, nb_beams, dim=0).to(device)
    
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
        
        logger.debug(generation_input.prompts)
        
        model_input = self.prepare_model_inputs(generation_input.prompts).to(self.get_device())
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=128,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            final_answers = self.get_final_generation(model_input['input_ids'], generated)
            del model_input
            return final_answers
    
    def beam_search(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):
        """
        Sends the `generation_input` to the model using beam search decoding

        Args:
            generation_input: Object containing the prompts to send to the model
            generation_config: Configuration object guiding the generation
        """

        # TODO : Implement batched note answering
        # Right now, questions can be batched, but only according to a single clinical note

        batch_size = len(generation_input.prompts)

        ontology_beam_scorer = OntologyBeamScorer(
            config=OntologyBeamScorerConfig(
                tokenizer=self.tokenizer,
                annotator=self.annotator,
                snomed=self.snomed,
                generation_input=generation_input,
                generation_config=generation_config
            ),
            batch_size=batch_size,
            device=self.get_device(),
        )

        prompts_tokenized = self.prepare_model_inputs(generation_input.prompts)        

        prompts_tokenized['input_ids'] = self.prepare_input_for_beam_search(
            prompts_tokenized['input_ids'], 
            device=self.get_device(),
            nb_beams=generation_config.nb_beams
        )

        prompts_tokenized['attention_mask'] = self.prepare_input_for_beam_search(
            prompts_tokenized['attention_mask'], 
            device=self.get_device(),
            nb_beams=generation_config.nb_beams
        )
        
        self.model.eval()
        with torch.no_grad():
            max_length = prompts_tokenized['input_ids'].shape[-1] + generation_config.max_new_tokens

            config_dict = {
                'num_beams': generation_config.nb_beams,
                'num_beam_groups': generation_config.nb_beam_groups,
                'do_sample': False,
                'diversity_penalty': generation_config.diversity_penalty,
                'num_return_sequences': 1,
                'max_length': max_length,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            hf_gen_config, _ = self.model._prepare_generation_config(None, **config_dict)
            
            logits_processor = LogitsProcessorList([])
            logits_warper = self.model._get_logits_processor(
                generation_config=hf_gen_config,
                input_ids_seq_length=prompts_tokenized['input_ids'].shape[-1],
                encoder_input_ids=prompts_tokenized['input_ids'],
                prefix_allowed_tokens_fn=None,
                logits_processor=[],
            )

            generations = self.model._group_beam_search(
                **prompts_tokenized,
                beam_scorer=ontology_beam_scorer,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=StoppingCriteriaList([
                    MaxLengthCriteria(max_length=max_length),
                ]),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            final_answers = self.get_final_generation(prompts_tokenized['input_ids'], generations)

            del prompts_tokenized
            return final_answers

    def generate(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):
        """
        Generates tokens based on a set of inputs

        Args:
            generation_input: Object containing the inputs to send to the model
            generation_config: Configuration object guiding the generation
        """
        if not generation_config.use_beam_search:
            return self.greedy_search(generation_input)
        return self.beam_search(generation_input, generation_config=generation_config)


BASE_PROMPT_TEMPLATE="""Here is a clinical note about a patient : 
-------------------
{clinical_note}
-------------------
In a short sentence, extract the information that is related to the "{label}" medical concept from the clinical note. If the concept is not mentioned in the note, respond with 'N/A'.
"""

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
        constrained_model: OntologyConstrainedModel, 
        snomed: Snomed, 
        annotator: Annotator, 
        template: OntologyPromptTemplate = OntologyPromptTemplate()
    ):
        
        """
        Args:
            constrained_model: reference to an aces model used to question the model
            snomed: reference to the snomed ontology 
            annotator: reference to the UMLSAnnotator.
        """
        
        self.constrained_model = constrained_model
        self.snomed = snomed
        self.annotator = annotator
        self.template = template

        self.attributes = []
        self.attributes_by_id = []

        # self.full_exclude_ids = set([self.snomed.base_class.id, '362981000', '123037004', '276339004', '106237007'])
        self.exclude_ids = set(['362981000', '444677008', '419891008', '276339004', '106237007'])

        self.current_note_id = 0

    def get_ancestors_adjusted_frequencies(self, frequencies: Dict[str, float]):
        """
        Adjusts frequencies to favor more general concepts first (generic term will probably
        have less ancestors). Modifies the score according to the following formula :
        new_score = 0.75 * old_score - nb_ancestors * 0.25
        """
        adjusted_frequencies = dict()
        for elem in frequencies.items():
            id, count = elem

            if not self.snomed.is_id_valid(id) or id in self.exclude_ids:
                continue
            else:
                # We want to favor more general concepts first 
                # A generic term will probably have less ancestors
                ancestors = self.snomed.get_ancestors_of_id(id, return_list=True)
                nb_ancestors = max(1, len(ancestors))
                new_score = (0.75 * count - nb_ancestors * 0.25)
                adjusted_frequencies[id] = new_score
        return adjusted_frequencies

    def get_all_ancestors_of_ids(self, ids: List[str]):
        """
        Returns a list containing all ancestors of all ids. If a concept is the ancestor
        of multiple ids, it will be present multiple times in the class.

        Args:
            ids: List of SNOMED ids

        Returns:
        All ancestors of all ids in a list (same id can be present multiple times)
        """
        
        all_ids = []
        for snomed_id in ids:
            ancs = self.snomed.get_ancestors_of_id(snomed_id, return_list=True)
            ancestor_included = all(map(lambda x: x not in self.exclude_ids, ancs))
            to_add = len(ancs) > 0 and ancestor_included
            if to_add:
                all_ids.append(ancs[0])
                all_ids.append(snomed_id)
        return all_ids
    
    def get_most_frequent_concepts(self, clinical_note, top_n):
        """
        Returns the `top_n` most frequent concepts present in the clinical note. It will
        tag the clinical note and then compute all ancestors of these 
        """
        
        concept_ids = self.annotator.annotate(clinical_note, return_ids_only=True) 
        all_ids = self.get_all_ancestors_of_ids(concept_ids)

        frequencies = Counter(all_ids)
        frequencies.pop(Snomed.BASE_CLASS_ID, 0)
        
        frequencies = self.get_ancestors_adjusted_frequencies(frequencies)

        if top_n == -1:
            top_n = len(frequencies)
        
        most_common_concepts = Counter(frequencies).most_common(top_n)
        return list(map(lambda x: x[0], most_common_concepts))
    
    def start_multiple(self, clinical_notes: List[str], top_n=5, batch_size=1, generation_config: GenerationConfig = GenerationConfig()):
        """
        Prompts a model on multiple clinical notes

        Args:
            notes: Clinical notes used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note
            generation_config: Configuration guiding the model's generation
        
        Returns:
        Tuple of dictionaries where the first dictionary contains {concept_id: extraction} and the 
        second dictionary contains {concept_label: extraction}
        """
        self.attributes.clear()
        self.attributes_by_id.clear()

        for i, note in enumerate(clinical_notes):
            self.current_note_id = i
            self.attributes_by_id.append({})
            self.attributes.append({})
            self.start(note, top_n=top_n, batch_size=batch_size, generation_config=generation_config)

        return self.attributes_by_id, self.attributes
    
    def start(
        self, 
        clinical_note: str, 
        top_n: int = 5, 
        batch_size: int = 1, 
        generation_config: GenerationConfig = GenerationConfig()
    ):
        """
        Prompts a model on a single clinical note

        Args:
            notes: Clinical note used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note
            generation_config: Configuration guiding the model's generation
        
        Returns:
        Tuple of dictionaries where the first dictionary contains {concept_id: extraction} and the 
        second dictionary contains {concept_label: extraction}
        """ 
        # stack = []
        # stack.append(self.snomed.base_class.id)

        most_frequent_concepts = self.get_most_frequent_concepts(clinical_note, top_n=top_n)

        if len(most_frequent_concepts) == 0:
            return
        
        logger.debug(f'Number of concepts extracted : {len(most_frequent_concepts)}')
        logger.debug(f'Most frequent concepts : {list(map(lambda x: x.label, self.snomed.convert_ids_to_classes(most_frequent_concepts)))}')
        
        for i in range((len(most_frequent_concepts) // batch_size) + 1):
            start = i * batch_size
            end = min(len(most_frequent_concepts), (i + 1) * batch_size)
            concept_ids = most_frequent_concepts[start:end]
            self.extract_attribute(clinical_note, concept_ids, generation_config=generation_config)

        # iteration = 1
        # while len(stack) > 0:
            
        #     start = max(0, (iteration - 1) * batch_size)
        #     end = min(len(most_frequent_concepts), iteration * batch_size)
        #     current_node_ids = most_frequent_concepts[start:end]
        #     self.extract_attribute(clinical_note, current_node_ids, generation_config=generation_config)
        #     iteration += 1
        #     if iteration * batch_size > len(most_frequent_concepts):
        #         break
        #     else:
        #         continue
    
    def extract_attribute(
        self, 
        clinical_note: str, 
        concept_ids: List[str], 
        generation_config: GenerationConfig = GenerationConfig()
    ):
        """
        Performs extraction step on a clinical note based on certain concepts from the ontology. It 
        then stores the extractions in `self.attributes_by_id` and `self.attributes`

        Args:
            clinical_note: Clinical note from which the information linked to concepts must be extracted
            concept_ids: Concept ids present in the clinical note guiding the extraction phase
            generation_config: Configuration guiding the model's generation
        """
        
        if len(concept_ids) == 0:
            return

        # Asking the model
        prompts = self.create_prompts(clinical_note, concept_ids)

        generation_input = GenerationInput(prompts=prompts, clinical_notes=[clinical_note] * len(prompts), concept_ids=concept_ids)
        answers = self.constrained_model.generate(generation_input, generation_config)

        if logger.level > 0:
            for answer in answers:
                logger.debug(f'\n{Color.RED}[Answer]{Color.OFF} : {answer.strip()}')
        
        # Storing answers
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

            if logger.level > 0:
                prompt = self.template.question_template.format_map({
                    'clinical_note': 'clinical note',
                    'label': label,
                    'properties': properties
                })
                logger.debug(f'\n{Color.CYAN}[Asking]{Color.OFF} : {prompt}')
                
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

            label = self.snomed.get_label_from_id(concept_id)

            if len(answer.strip()) > 0 and valid_answer:
                self.attributes[self.current_note_id][label] = answer
                self.attributes_by_id[self.current_note_id][concept_id] = answer
            else:
                self.attributes[self.current_note_id][label] = 'N/A'
                self.attributes_by_id[self.current_note_id][concept_id] = 'N/A'



class OntologyBasedAnalyzer:
    """
    Responsable for the pruning and merging phase
    """
    
    def __init__(
        self, 
        result, 
        annotator: Annotator, 
        snomed: Snomed, 
        tokenizer,
        notes_column: str = 'notes',
        attributes_column: str = 'attributes',
        attributes_by_id_column: str = 'attributes_by_id',
        clinical_note_processor = None
    ):
        self.result = result
        self.annotator = annotator
        self.snomed = snomed
        self.tokenizer = tokenizer
        self.clinical_note_processor = clinical_note_processor
        
        self.notes_column = notes_column
        self.attributes_column = attributes_column
        self.attributes_by_id_column = attributes_by_id_column

        self.parse_data()
        
    def parse_data(self):
        
        self.raw_notes = self.result[self.notes_column]
        if self.clinical_note_processor is not None:
            self.notes = self.result[self.notes_column].apply(self.clinical_note_processor)
        # else:
            # self.notes = preprocess_clinical_notes(self.result[self.notes_column], self.tokenizer)
        
        self.attributes = ast.literal_eval(self.result[self.attributes_column])
        self.attributes_by_id = ast.literal_eval(self.result[self.attributes_by_id_column])
        self.exclude_ids = set([self.snomed.base_class.id, '362981000', '123037004', '276339004', '106237007', '444677008'])

        self.preprocess_data()

    def preprocess_data(self):
        for i, attr in enumerate(self.attributes):            
            for k, v in attr.items():
                real_value = v.replace('. \n', '')
                real_value = real_value.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                if real_value.find('The clinical note does not') >= 0:
                    real_value = 'N/A'
                attr[k] = real_value

        for i, attr in enumerate(self.attributes_by_id):            
            for k, v in attr.items():
                real_value = v.replace('. \n', '')
                real_value = real_value.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                if real_value.find('The clinical note does not') >= 0:
                    real_value = 'N/A'
                attr[k] = real_value


    def show_results(self, attributes = None, show_na = False):
                
        if attributes is None:
            attributes = self.attributes
            
        if isinstance(attributes, dict):
            for k, v in attributes.items():
                    real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                    can_show = show_na or (not show_na and 'N/A' not in real_value)
                    can_show &= len(real_value) > 0 
                    if can_show:
                        print(f'\x1b[31m{k}\x1b[0m', ' : ', real_value)
            return
        
        for i, attr in enumerate(attributes):            
            if len(attr) > 0:
                print('=' * 10, f'Note {i}', '=' * 10)
            for k, v in attr.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    print(f'\x1b[31m{k}\x1b[0m', ' : ', real_value)

    def get_printable_results(self, attributes = None, show_na = False):
        if attributes is None:
            attributes = self.attributes
        
        out = ''
        if isinstance(attributes, dict):
            for k, v in attributes.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    out += k + ':' + real_value + '\n'
            return out
        

        for i, attr in enumerate(attributes):
            if len(attr) > 0:
                out += '=' * 10 + f' Note {i} ' + '=' * 10 + '\n'
            for k, v in attr.items():
                real_value = v.strip().replace('\n\n', '\n').replace('\n- ', ', ')
                can_show = show_na or (not show_na and 'N/A' not in real_value)
                can_show &= len(real_value) > 0 
                if can_show:
                    out += k + ':' + real_value + '\n'
        return out
    
    def filter_results(self, x: AnnotationMatch):
        ancestors = self.snomed.get_ancestors_of_id(x.snomed_id, return_set=True)
        ancestors.add(x.snomed_id)
        return len(ancestors.intersection(self.exclude_ids)) == 1 # Only the base class

    def get_annotation_overlap(self, current_content, new_content):
        current_content_concepts = self.annotator.annotate(current_content, result_filter=self.filter_results)
        new_content_concept = self.annotator.annotate(new_content, result_filter=self.filter_results)
        current_content_concepts = set(map(lambda x: x.snomed_id, current_content_concepts))
        new_content_concept = set(map(lambda x: x.snomed_id, new_content_concept))

        if len(new_content_concept) == 0:
            return 0
        
        return len(current_content_concepts.intersection(new_content_concept)) / len(new_content_concept)

    def get_merged_attributes(self, overlap_threshold: float = 0.5):
        merged_attributes = {}
        
        # Merging all notes together
        for attribute in self.attributes_by_id:

            for id, value in attribute.items():
                if 'N/A' in value:
                    continue

                if id not in merged_attributes:
                    merged_attributes[id] = '\n\t- ' + value
                else:
                    current_content = merged_attributes[id]
                    new_content = value

                    should_merge = True
                    for single_current_content in current_content.split('\n\t- '):
                        # When content is added it is appended. We need to split to analyze
                        # each content separately 
                        if len(single_current_content) == 0:
                            continue
                        
                        if single_current_content.lower().strip() == new_content.lower().strip():
                            # Current content is identical to new content
                            should_merge = False
                            break

                        nb_overlap_concepts = self.get_annotation_overlap(single_current_content, new_content)
                        if nb_overlap_concepts >= overlap_threshold:
                            # The concept is already evoqued by a content that is already present
                            should_merge = False
                            break

                    if should_merge:
                        merged_attributes[id] = current_content + '\n\t- ' + new_content

        result = {}
        for key, value in merged_attributes.items():
            if not any(sentence_bleu([splitted_value], value, (0.5, 0.5)) > 0.90 # TODO : Put this value as a parameter
                    for existing_value in result.values() for splitted_value in existing_value.split('\n\t- ')):
                result[key] = value
        return result

    
    def get_filtered_attributes(self, attributes: Dict[str, str], filter_attributes: List[str], ancestor_level=2):
        """
        Computes the intersection of two attribute list by taking all attributes from `attributes` 
        that are are least an `ancestor_level` distance from attributes in `filter_attributes`.
        """
        # print('Number of attributes extracted : ', len(attributes))
        filtered_attributes = dict()
        for id, value in attributes.items():
            ancestors_id: set = self.snomed.get_ancestors_of_id(id, return_set=True)
            for filtered in filter_attributes:
                ancestors: set = self.snomed.get_ancestors_of_id(filtered, return_set=True)
                if len(ancestors_id.intersection(ancestors)) >= ancestor_level:
                    # Two classes are related if they have more than `ancestor_level` common ancestors
                    filtered_attributes[id] = value
        # print('Number of attributes kept : ', len(filtered_attributes))
        return filtered_attributes

    def get_adjusted_frequencies(self, initial_frequencies: Dict[str, str]):
        """
        Adjusts the frequencies of a frequency dictionary by taking into account the number of ancestor
        to favor more precise concepts
        """
        adjusted_frequencies = dict()
        for elem in initial_frequencies.items():
            id, count = elem

            if id not in self.snomed.id_to_classes or id in self.exclude_ids:
                continue
            else:
                # We want to favor more general concepts in this case 
                # A generic term will probably have less ancestors
                ancestors = self.snomed.get_ancestors_of_id(id, return_list=True)
                nb_ancestors = max(1, len(ancestors))
                
                new_score = 0.95 * count - 0.05 * nb_ancestors
                adjusted_frequencies[id] = new_score
        return adjusted_frequencies


    def counter_list_to_dict(self, tuple_list: List):
        """
        Converts a list of tuples in the form `(elem, count)` to a dictionary in the form `dict[elem] = count`
        """
        filtered_most_common = dict()
        for common in tuple_list:
            id, count = common
            # if id not in self.exclude_ids:
            filtered_most_common[id] = count
        return filtered_most_common

    def adapt_to_domain(
        self, 
        domain_frequencies: DomainClassFrequency,
        merged: bool = False, 
        top_n: int = 50, 
        ancestor_level: int = 1, 
        adjust_frequencies: bool = True,
        overlap_threshold: float = 0.5 # For merging
    ):
        if adjust_frequencies:
            domain_frequencies.frequencies = self.get_adjusted_frequencies(domain_frequencies.frequencies)

        if top_n == -1:
            top_n = len(domain_frequencies.frequencies)

        filtered_most_common = domain_frequencies.get_most_common(top_n=top_n, exclude_set=self.exclude_ids, snomed=self.snomed)
        filtered_most_common = [x[0] for x in filtered_most_common] # [(id1, count1), (id2, count2), ...] -> [id1, id2, ...]
        if not merged:
            final_domain_attributes = []
            for attribute in self.attributes_by_id:
                domain_attributes = self.get_filtered_attributes(attribute, filtered_most_common, ancestor_level=ancestor_level)
                final_domain_attributes.append(domain_attributes)
            
            return list(map(lambda x: self.convert_ids_to_labels(x), final_domain_attributes)), final_domain_attributes

        merged_attributes = self.get_merged_attributes(overlap_threshold=overlap_threshold)
        domain_attributes = self.get_filtered_attributes(merged_attributes, filtered_most_common, ancestor_level=ancestor_level)
        return self.convert_ids_to_labels(domain_attributes), domain_attributes
    
    def convert_ids_to_labels(self, attributes):
        labeled_attributes = {}
        for k, v in attributes.items():
            labeled_attributes[self.snomed.get_class_from_id(k).label] = v
        return labeled_attributes


class BHCOntologyBasedVerbalizer:
    """
    Applies the pruning + verbalizer stage to an unstructured input
    """

    def __init__(
        self, 
        output_path: str,
        snomed: Snomed,
        annotator: Annotator,
        model,
        tokenizer,
    ) -> None:
        self.output_path = output_path
        self.snomed = snomed
        self.annotator = annotator
        self.model = model
        self.tokenizer = tokenizer

        if os.path.exists(self.output_path):
            self.df = pd.read_csv(self.output_path)
        else:
            self.df = pd.DataFrame(
                [], 
                columns=[
                    'text', 
                    'summary', 
                    'structured', 
                    'unstructured',
                    'prompt'
                ]
            )

    def ask_model_batch(self, x: list[str], max_new_tokens=1536, limit_tokens = 14000):
        clear_gpu_cache()
        chat_template = ChatTemplate(self.tokenizer)    
        prompts = list(map(lambda y: chat_template.add_user_entry(y), x))        
        model_input = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        print('Input Length : ', model_input['input_ids'].shape)
        if limit_tokens > 0 and model_input['input_ids'].shape[-1] >= limit_tokens:
            return ''
        
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=max_new_tokens,
            )
            # Only retrieve the newly generated tokens
            new_tokens = generated[:, model_input['input_ids'].shape[-1]:]

            results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            del model_input
            return results


    def start(
        self,
        structured: pd.DataFrame,
        pre_prompt: str,
        post_prompt: str,
        extract: bool = False,
        prune: bool = False,
        merged: bool = False,
        domain_frequencies: DomainClassFrequency = None,
        top_n_concepts: int = 100,
        ancestor_level: int = 2,
        text_column: str = 'text',
        summary_column: str = 'summary',
        debug: int = 0,
    ):
        initial = len(self.df)
        for i in tqdm(range(initial, len(structured)), total=len(structured), initial=initial):
        # for i in range(initial, len(structured)):
            
            current_result = structured.iloc[i]
            analyzer = OntologyBasedAnalyzer(
                current_result, 
                self.annotator, 
                self.snomed, 
                self.tokenizer,
                notes_column='notes',
                attributes_column='attributes',
                attributes_by_id_column='attributes_by_id'
            )
            id = current_result['id']
            summary = current_result[summary_column]
            text = current_result[text_column]

            # Method to adapt the text to the correct length of the summary
            # suggested_length = max(100, 100 * int(len(nltk.word_tokenize(summary)) / 100))

            if not extract:
                # Normal generation
                prompt = pre_prompt + text + post_prompt
            elif extract and not prune:
                # Extraction + Pruning + Verbalizer
                prompt = analyzer.get_printable_results()
                prompt = pre_prompt + prompt + post_prompt
            else:
                domain, _ = analyzer.adapt_to_domain(
                    domain_frequencies, 
                    merged=merged, 
                    top_n=top_n_concepts, 
                    ancestor_level=ancestor_level
                )
                prompt = analyzer.get_printable_results(domain)
                prompt = pre_prompt + prompt + post_prompt
            
            if len(prompt.strip()) - len(pre_prompt) - len(post_prompt) <= 500:
                print('Very few attributes were extracted : ', prompt)
            
            # prompt = prompt.replace('400-word', f'{suggested_length}-word')
            if debug >= 2:
                print(prompt)
            
            unstructured = self.ask_model_batch([prompt])
            if debug >= 1:
                print(unstructured)

            new_row = {
                'id': id,
                'text': text, 
                'summary': summary,
                'structured': current_result.attributes_by_id,
                'unstructured': unstructured,
                'prompt': prompt
            }

            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self.df.to_csv(self.output_path)


class DomainOntologyBasedVerbalizer:
    """
    Applies the pruning + merging + verbalizer stage to an unstructured input
    """

    DOMAIN_HEADER = '<|domain|>'

    def __init__(
        self, 
        output_path: str,
        domain_analysis: Dict[str, DomainClassFrequency],
        snomed: Snomed,
        annotator: Annotator,
        model,
        tokenizer,
    ) -> None:
        self.output_path = output_path
        self.domain_analysis = domain_analysis
        self.snomed = snomed
        self.annotator = annotator
        self.model = model
        self.tokenizer = tokenizer

        if os.path.exists(self.output_path):
            self.df = pd.read_csv(self.output_path)
        else:
            self.df = pd.DataFrame(
                [], 
                columns=[
                    'text', 
                    'summary', 
                    'structured', 
                    'unstructured',
                    'domain'
                ]
            )

    def ask_model_batch(self, x: list[str], max_new_tokens=1536):
        clear_gpu_cache()
        chat_template = ChatTemplate(self.tokenizer)    
        prompts = list(map(lambda y: chat_template.add_user_entry(y), x))        
        model_input = self.tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=max_new_tokens,
            )
            # Only retrieve the newly generated tokens
            new_tokens = generated[:, model_input['input_ids'].shape[-1]:]

            results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            del model_input
            return results


    def start(
        self,
        structured: pd.DataFrame,
        pre_prompt: str,
        post_prompt: str,
        normal: bool = True,
        top_n_concepts: int = 100,
        ancestor_level: int = 4,
        text_column: str = 'text',
        summary_column: str = 'summary',
        domain_filter: List[str] = None
    ):
        initial = len(self.df) // len(self.domain_analysis)
        for i in tqdm(range(initial, len(structured)), total=len(structured), initial=initial):
            
            current_result = structured.iloc[i]
            analyzer = OntologyBasedAnalyzer(
                current_result, 
                self.annotator, 
                self.snomed, 
                self.tokenizer,
                notes_column='notes',
                attributes_column='attributes',
                attributes_by_id_column='attributes_by_id'
            )
            summary = current_result[summary_column]
            text = current_result[text_column]

            for domain, dcf in self.domain_analysis.items():

                if domain_filter is not None and domain not in domain_filter:
                    continue
                print(f'Adapting to {domain} domain')

                if normal:
                    prompt = pre_prompt + text + post_prompt
                else:
                    domain_results, _ = analyzer.adapt_to_domain(
                        dcf,
                        top_n=top_n_concepts, 
                        ancestor_level=ancestor_level
                    )
                    prompt = analyzer.get_printable_results(domain_results)
                    prompt = pre_prompt + prompt + post_prompt
                
                prompt = prompt.replace(DomainOntologyBasedVerbalizer.DOMAIN_HEADER, domain)
                unstructured = self.ask_model_batch([prompt])

                new_row = {
                    'text': text, 
                    'summary': summary,
                    'structured': current_result.attributes_by_id,
                    'unstructured': unstructured,
                    'domain': domain
                }

                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                # self.df_results = pd.DataFrame(predictions, columns=['clinical_notes', 'aces_prediction', 'aces_bhc_prediction', 'normal_prediction', 'target'])
                self.df.to_csv(self.output_path)
