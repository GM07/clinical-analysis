

from transformers import BeamSearchScorer
from rouge_score import rouge_scorer

from typing import Dict, List, Optional, Tuple, Union, ClassVar
from dataclasses import dataclass
import time
import torch

from src.generation.generation import GenerationInput
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed

@dataclass
class GenerationConfig:
    """
    Class controlling a model's generation. Dictates whether beam search will be used and weather the ontology-based beam search
    will be used.
    """

    max_length: int = 2048
    max_new_tokens: int = 128
    use_beam_search: bool = False
    normal_beam_search: bool = False
    nb_beams: int = 10
    nb_beam_groups: int = 2
    nb_beam_hyps_to_keep: int = 1
    window_size: int = 5
    diversity_penalty: float = 1.0
    
    # Weights associated with the original beam scores and the boost scores.
    # The final scores will be `score_weights[0]` * original_beam_scores + `score_weights[1]` * custom_scores
    score_weights: Tuple[float, float] = (0.0, 1.0)

    # Corresponds to H_bf, P_bf and G_bf
    score_boost_factors = [3.0, 1.0, 10.0]

    # The exclude_ids list contains classes that are not entirely linked to concepts
    # but more like values or environments. This is usually what is present in the 
    # answer. They should be excluded in the prompting process, but are important in
    # the decoding process
    exclude_ids: ClassVar[set[str]] = set([]) # set(['362981000', '419891008', '106237007'])
    
# group_size = nb_beams / nb_beam_groups
# input_size = batch_size * group_size

class OntologyBeamScorerConfig:
    """
    Configuration class for the ontology beam scorer
    """
    def __init__(
        self, 
        tokenizer, 
        annotator: Annotator, 
        snomed: Snomed, 
        generation_input: GenerationInput,
        generation_config: GenerationConfig
    ) -> None:
        """
        Args:
            tokenizer: Tokenizer used by the model (used to decode the beams and annotate)
            annotator: Annotator to retrieve to concept ids from the beams
            snomed: Snomed ontology object used to retrieve the ancestors of a concept
            generation_input: Input of the generation containing the prompts, the clinical notes and the concept ids
            generation_config: Configuration guiding the model's generation in the beam search algorithm
        """
        self.tokenizer = tokenizer
        self.annotator = annotator
        self.snomed = snomed
        self.generation_input = generation_input
        self.generation_config = generation_config

class OntologyBeamScorer(BeamSearchScorer):
    """
    Ontology-based beam search scorer
    """

    def __init__(
        self,
        config: OntologyBeamScorerConfig,
        batch_size: int, 
        device: torch.device, 
        length_penalty: Optional[float] = 1.0, 
        do_early_stopping: Optional[Union[bool, str]] = False, 
    ):
        """
        Args:
            config: Configuration of the algorithm
        
        For other arguments, @see `BeamSearchScorer`
        """
        super().__init__(
            batch_size, 
            config.generation_config.nb_beams, 
            device, 
            length_penalty, 
            do_early_stopping, 
            config.generation_config.nb_beam_hyps_to_keep, 
            config.generation_config.nb_beam_groups, 
            config.generation_config.max_length
        )
        self.batch_size = batch_size
        self.config = config
        self.nb_tokens_generated = self.config.generation_config.window_size // 2

        self.time_to_next_process = -1
        self.nb_times_process_called = 0
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)


    def get_base_class_id_from_index(self, index):
        """
        Retrieves the associated base class of a prompt given an 
        in index from a batched, beamed input_id tensor

        Args:
            index: Index in the batched, beamed tensor

        Returns:
        Associated concept id linked to the prompt
        """
        batch_index = index % self.batch_size
        return self.config.generation_input.concept_ids[batch_index]

    def get_clinical_note_from_index(self, index):
        """
        Retrieves the associated base class of the query given an 
        in index from a batched, beamed input_id tensor

        Args:
            index: Index in the batched, beamed tensor

        Returns:
        Associated clinical note linked to the prompt
        """
        batch_index = index % self.batch_size
        return self.config.generation_input.clinical_notes[batch_index]

    def get_hierarchy_beam_boost(self, base_class_id: str, detected_class_id: str):
        """
        Computes the hierarchy score of the beam

        Args:
            base_class_id: Id of base class related to the prompt
            detected_class_id: Id of the concept detected in the beam

        Returns:
        Hierarchy score
        """

        if detected_class_id not in self.config.snomed.id_to_classes:
            return 0

        if detected_class_id == base_class_id:
            return 1.0

        parents = self.config.snomed.get_ancestors_of_id(detected_class_id, return_set=True)
        parents_in_exclusion_set = parents.intersection(GenerationConfig.exclude_ids)
        if len(parents_in_exclusion_set) > 0:
            return 0.0
        
        if base_class_id not in parents:
            # We don't want concepts from other branches
            return -1.0
        
        return 1.0

    def get_groundedness_beam_boost(self, index: int, context: str):
        """
        Compares the current context beam to the clinical notes to encourage
        beams to ressemble the formulation of the clinical notes.

        Args:
            index: Index of the beam.
            context: Context of the current beam
        """
        # We use every sentence of the clinical notes as a reference
        references = self.get_clinical_note_from_index(index).split('. ')

        # Compute ROUGE-2
        beam_boost = max(map(lambda x: self.rouge_scorer.score(context, x)['rouge2'].precision, references))
        return beam_boost

    def get_properties_beam_boost(self, base_class_id: str, detected_class_id: str, context: str):
        """
        Computes the property score of the beam

        Args:
            base_class_id: Id of base class related to the query
            detected_class_id: Id of the concept detected in the beam
            context: Context where `detected_class_id` was detected (clinical note)
        """
        
        properties = self.config.snomed.get_restriction_properties_of_id(base_class_id)
        if len(properties) == 0:
            return 0
        
        property_score = 0

        # Direct property link in ontology
        for property in properties:
            for k, v in property.ids_to_ids.items():
                if k == detected_class_id or v == detected_class_id:
                    # Detected concept id is directly linked to the base class id
                    property_score += 1 / (len(property.ids_to_ids))

                detected_class_ancestors = self.config.snomed.get_ancestors_of_id(detected_class_id, return_set=True)
                if v in detected_class_ancestors:
                    property_score += 1 / (len(property.ids_to_ids))
                
        # Indirect property link : Add the rouge score between all property values and the context
        current_property_knowledge = ' '.join(map(lambda x: x.get_value(), properties))        
        rouge_score = self.rouge_scorer.score(context, current_property_knowledge)['rouge2'].precision

        property_score += rouge_score

        # We divide by two since the direct property link is between 0 and 1
        # and the indirect property link is between 0 and 1. Thus the max value
        # of the property score is 2
        return property_score / 2 

    def get_beam_boost(self, input_ids, group_index):
        """
        Computes the beam score according to the ontology and the generated beam content

        Args:
            input_ids: Input ids of all the beams in the group
            group_index: Index of the group in the diverse beam search algorithm (only used for debugging purposes)
        """
        context_size = int(2 * self.config.generation_config.window_size)
        decoded_next_tokens = self.config.tokenizer.batch_decode(input_ids[:, -context_size:])
        batched_annotations = self.config.annotator.batch_annotate(decoded_next_tokens, return_ids_only=True)
        
        scores = []
        for i, annotations in enumerate(batched_annotations):
            base_class_id = self.get_base_class_id_from_index(i)
            decoded_context = decoded_next_tokens[i]

            avg_hierarchy_score = 0
            avg_property_score = 0
            for snomed_id in annotations:
                avg_hierarchy_score += self.get_hierarchy_beam_boost(base_class_id, snomed_id)
                avg_property_score += self.get_properties_beam_boost(base_class_id, snomed_id, decoded_context)

            avg_hierarchy_score /= max(1, len(annotations))
            avg_property_score /= max(1, len(annotations))
            groundedness_score = self.get_groundedness_beam_boost(i, decoded_context)

            freq_adapted_score = \
                  self.config.generation_config.score_boost_factors[0] * avg_hierarchy_score \
                + self.config.generation_config.score_boost_factors[1] * avg_property_score \
                + self.config.generation_config.score_boost_factors[2] * groundedness_score

            scores.append(freq_adapted_score)
        return scores
    
    def compute_new_beam_scores(self, current_scores, ontology_scores):
        """
        Given the current beam scores and the new ontology scores, computes a weighted average
        between the current scores and the ontology scores

        Args:
            current_scores: Current scores of the beams
            ontology_scores: New beam scores based on ontology

        Returns:
        Weighted average of current scores and ontology scores
        """
        ontology_scores = torch.tensor(
            ontology_scores, 
            device=current_scores.device, 
            dtype=torch.float32
        )
        ontology_scores = torch.nn.functional.log_softmax(ontology_scores, dim=-1)

        weights = self.config.generation_config.score_weights
        final_results = (weights[0] * current_scores + weights[1] * ontology_scores) / (weights[0] + weights[1])

        del ontology_scores
        return final_results

    def process(
        self, 
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        @see `BeamSearchScorer.process`
        """
        if self.time_to_next_process < 0:
            self.time_to_next_process = time.time()
        else:
            self.time_to_next_process = time.time()
        
        results = super().process(input_ids, next_scores, next_tokens, next_indices, pad_token_id, eos_token_id, beam_indices, group_index, decoder_prompt_len)

        if self.config.generation_config.normal_beam_search:
            return results

        self.nb_times_process_called += 1
        modifying_scores: bool = self.nb_tokens_generated >= self.config.generation_config.window_size
        if modifying_scores:
            old_beam_scores = results['next_beam_scores']
            boost_factors = self.get_beam_boost(input_ids, group_index=group_index)
            results['next_beam_scores'] = self.compute_new_beam_scores(old_beam_scores, boost_factors)

            # The beam search scorer will be called for each group.
            # Thus, every `window_size` tokens, we need to process the next
            # `nb_group` calls to this function
            if group_index >= self.config.generation_config.nb_beam_groups - 1:
                self.nb_tokens_generated = 1 # While we modified the scores, a token was processed
        else:
            # print('not modifying')
            if group_index >= self.config.generation_config.nb_beam_groups - 1:
                # We increase the number of tokens generated only if we have processed
                # every group since this function is called for every group for every token
                self.nb_tokens_generated += 1

        return results
    
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.LongTensor]:
        """
        @see `BeamSearchScorer.finalize`
        """
        return super().finalize(input_ids, final_beam_scores, final_beam_tokens, final_beam_indices, max_length, pad_token_id, eos_token_id, beam_indices, decoder_prompt_len)
