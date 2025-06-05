from typing import List
import logging
import types

import torch
from accelerate import Accelerator

from transformers.generation.configuration_utils import GenerationConfig as HFGenerationConfig
from transformers.generation.utils import GenerationMixin

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


    def prepare_model_inputs(self, prompts: List[str], system_prompt: str = None):
        """
        Prepares a list of prompts to be sent to the model by applying the chat template and tokenizing the input
        
        Args:
            prompts: List of prompts to send to the model
        """
        if self.apply_chat_template and self.tokenizer.chat_template is not None:
            prompts = self.chat_template.batched_single_user_entry(prompts, system_entry=system_prompt)
        
        model_input = self.tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=False, 
            pad_to_multiple_of=8,
            add_special_tokens=True # Modified for llama-medicine
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


    def greedy_search(self, generation_input: GenerationInput, generation_config: GenerationConfig) -> List[str]:
        """
        Sends the `generation_input` to the model using greedy search decoding

        Args:
            generation_input: Object containing the prompts to send to the model
            generation_config: Generation config of the model
        """

        model_input = self.prepare_model_inputs(generation_input.prompts, system_prompt=generation_input.system_prompt)
        model_input['input_ids'] = model_input['input_ids'].to(self.get_device())
        model_input['attention_mask'] = model_input['attention_mask'].to(self.get_device())

        self.normal_generate()

        hf_generation_config = HFGenerationConfig(
                # temperature=0,
                top_p=1,
                # top_k=-1,
                seed=42
        )

        with torch.no_grad():
            generated = self.model.generate(
                **model_input, 
                max_new_tokens=generation_config.max_new_tokens,
                generation_config=hf_generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            final_answers = self.get_final_generation(model_input['input_ids'], generated)
            del model_input
            return final_answers
    

    def group_beam_search(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):

        tokenized_inputs = self.prepare_model_inputs(generation_input.prompts, system_prompt=generation_input.system_prompt)
        input_ids = tokenized_inputs['input_ids'].to(self.get_device())
        attention_mask = tokenized_inputs['attention_mask'].to(self.get_device())
        hf_generation_config = HFGenerationConfig(
            top_p=1,
            num_beams=generation_config.nb_beams,  # Number of beams for beam search
            num_return_sequences=1,  # Return all beams
            num_beam_groups=generation_config.nb_beam_groups,
            diversity_penalty=generation_config.diversity_penalty,
            seed=42
        )

        with torch.no_grad():
            if generation_config.normal_beam_search:
                self.normal_generate()
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_generation_config,
                    max_new_tokens=generation_config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
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
                    device=self.get_device()
                )
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_generation_config,
                    max_new_tokens=generation_config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    beam_scorer=ontology_beam_scorer
                )
            generated_tokens = generation_output
            final_answers = self.get_final_generation(tokenized_inputs['input_ids'], generated_tokens)
            return final_answers

    def generate(self, generation_input: GenerationInput, generation_config: GenerationConfig = GenerationConfig()):

        if generation_config.use_group_beam_search:
            return self.group_beam_search(generation_input, generation_config)
        else:
            return self.greedy_search(generation_input, generation_config)


class OntologyPromptTemplate:

    def __init__(self, question_template: str = None):
        if question_template is None:
            self.question_template = BASE_PROMPT_TEMPLATE
        else:
            self.question_template = question_template
