from typing import Dict, List
from collections import Counter, defaultdict
import logging

from tqdm import tqdm

import torch
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria
from accelerate import Accelerator

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.ontology_beam_scorer import OntologyBeamScorer, OntologyBeamScorerConfig, GenerationInput, GenerationConfig
from src.generation.chat_template import ChatTemplate
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed
from src.generation.templates import BASE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

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
        if self.apply_chat_template and self.tokenizer.chat_template is not None:
            prompts = self.chat_template.batched_single_user_entry(prompts)
        
        logger.debug(prompts)

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
        
        model_input = self.prepare_model_inputs(generation_input.prompts).to(self.get_device())
        self.model.eval()

        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=128,
                temperature=None,
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
                'use_cache': False, # TODO : Change when transformers bug is fixed
                'max_length': max_length,
                'eos_token_id': self.tokenizer.eos_token_id,
                'bos_token_id': self.tokenizer.bos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id
            }
            hf_gen_config, model_kwargs = self.model._prepare_generation_config(None, **config_dict)
            kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
            self.model._prepare_special_tokens(hf_gen_config, kwargs_has_attention_mask, device=self.model.device)
            # print(hf_gen_config.__dir__())
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
                generation_config=hf_gen_config,
                synced_gpus=False,
                use_cache=False
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

        
        self.attributes = []
        self.attributes_by_id = []

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
        
        self.attributes = []
        self.attributes_by_id = []

        for i, note in enumerate(clinical_notes):
            self.current_note_id = i
            self.attributes_by_id.append({})
            self.attributes.append({})
            self.start(note, top_n=top_n, batch_size=batch_size, generation_config=generation_config)

            return self.attributes_by_id.copy(), self.attributes.copy()
    
    def start_dataset(self, clinical_notes: List[str], ids: List[str], top_n: int = 5):
        """
        Starts the extraction on a dataset
        """

        assert len(ids) == len(clinical_notes), 'The number of ids should be the same as the number of clinical notes'
        
        logger.info(f'Generating prompts for dataset')
        dataset = defaultdict(list)

        for id, clinical_note in tqdm(zip(ids, clinical_notes), total=len(ids)):
            most_frequent_concepts, frequencies = DomainClassFrequency.get_most_frequent_concepts(
                text=clinical_note, 
                snomed=self.snomed, 
                annotator=self.annotator, 
                top_n=top_n
            )
            prompts = self.create_prompts(clinical_note, most_frequent_concepts)
            dataset[id] = prompts

        return dataset

    def start(
        self, 
        clinical_note: str, 
        top_n: int = 5, 
        batch_size: int = 1, 
        generation_config: GenerationConfig = GenerationConfig()
    ):
        """
        Retrieves the most frequent concepts present in the clinical note and extracts them (or stores the prompt if `self.dataset_mode` is `True`)

        Args:
            notes: Clinical note used to extract information from
            top_n: Number maximal concepts to extract from each clinical note
            batch_size: Number of concepts to process in parallel per clinical note (not used if `self.dataset_mode` is `True`)
            generation_config: Configuration guiding the model's generation (not used if `self.dataset_mode` is `True`)
        
        Returns:
        Tuple of dictionaries where the first dictionary contains {concept_id: extraction} and the 
        second dictionary contains {concept_label: extraction}
        """ 
        most_frequent_concepts, frequencies = DomainClassFrequency.get_most_frequent_concepts(
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
        then stores the extractions in `self.attributes_by_id` and `self.attributes`

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

            label = self.snomed.get_label_from_id(concept_id)

            if len(answer.strip()) > 0 and valid_answer:
                self.attributes[self.current_note_id][label] = answer
                self.attributes_by_id[self.current_note_id][concept_id] = answer
            else:
                self.attributes[self.current_note_id][label] = 'N/A'
                self.attributes_by_id[self.current_note_id][concept_id] = 'N/A'
