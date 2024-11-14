import ast
from typing import List
import itertools

import pandas as pd

from src.model_registry import LoadingConfig, ModelRegistry
from src.generation.generation import BASE_PROMPT_TEMPLATE
from src.ontology.snomed import Snomed
from src.utils import run_inference

FACTUALITY_RUBRIC = f"""[Are the model's responses factually correct and well-supported by evidence?]
Score 1: The model's responses are mostly incorrect or based on unfounded information.
Score 2: The model sometimes provides factually correct responses, but inaccuracies are common.
Score 3: The model generally provides factually correct information, though some errors occur.
Score 4: The model often provides factually accurate information with only occasional minor errors.
Score 5: The model consistently provides responses that are factually correct and well-supported by evidence.""".strip()


RELEVANCE_RUBRIC = f"""[Are the model's responses relevant to the medical concept mentioned?]
Score 1: The model's answer is irrelevant to the medical concept and completely misses information that is related to the medical concept
Score 2: The model's short summary is mainly irrelevant, but mentions one or two things related to the medical concept mentioned
Score 3: The model's short summary is somewhat irrelevant, but contains key elements related to the concept mentioned
Score 4: The model's short summary is mainly relevant, but contains some elements that are not linked to the medical concept
Score 5: The model's short summary mentions everything related the the medical concept perfectly without missing any detail""".strip()

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{rubric}

###Feedback: """

class PrometheusPromptGenerator:
    """
    Evaluates extractions on clinical notes from the mimic dataset using the Prometheus model
    """

    def __init__(self, model_checkpoint: str, snomed_path: str, snomed_cache_path: str, loading_config: LoadingConfig = LoadingConfig()):
        """
        Args:
            model_checkpoint: Path to the HuggingFace checkpoint of the model Prometheus
            loading_config: Configuration object on how to load the model
        """
        self.model_checkpoint = model_checkpoint
        self.loading_config = loading_config
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path

        self.load()
    
    def load(self):
        """Loads the model and the ontology"""
        self.model, self.tokenizer = ModelRegistry.load_single_checkpoint(self.model_checkpoint, loading_config=self.loading_config)
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path, nb_classes=366771)

    def get_model_prompt(self, clinical_note, ontology_concept):
        """Reproduces the prompt used in the extraction step with a clinical note and an ontology concept"""
        return BASE_PROMPT_TEMPLATE.format(clinical_note=clinical_note, label=ontology_concept)

    def get_prometheus_prompt(self, clinical_note: str, ontology_concept: str, response_a: str, response_b: str, rubric: str):
        """
        Returns the prompt used by prometheus to evaluate the responses
        
        Args:
            clinical_note: Clinical note used in the extraction step
            ontology_concept: Ontology concept used in the extraction step
            response_a: Response of model A
            response_b: Response of model B
            rubric: Rubric used to evaluate the responses
        """
        if rubric not in ['factuality', 'relevance']:
            raise ValueError(f'The provided rubric ({rubric}) is not supported. Here are the supported rubrics : factuality, relevance')
        
        if rubric == 'factuality':
            rubric = FACTUALITY_RUBRIC
        else:
            rubric = RELEVANCE_RUBRIC

        return ABS_SYSTEM_PROMPT + '\n\n' + ABSOLUTE_PROMPT.format(
            instruction=self.get_model_prompt(clinical_note, ontology_concept),
            response_A=response_a,
            response_B=response_b,
            rubric=rubric
        )

    def generate_prompts(
        self, 
        results: pd.DataFrame, 
        rubric: str,
        output_file_path: str,
        clinical_note_column: str = 'TEXT', 
        clinical_note_id_column: str = 'ROW_ID', 
        result_columns: List[str] = ['normal', 'beam', 'constrained']
    ):
        """
        Will evaluate results using prometheus on a result file (previously generated from the function `Dataset.partitions_to_file`)

        Args:
            results: DataFrame generated by the function `Dataset.partitions_to_file`
            rubric: Rubric used to evaluate each sample
            output_file_path: Where to save the prompts
            clinical_note_column: Column in the DataFrame containing the clinical note for each sample
            clinical_note_id_column: Column in the DataFrame containing the clinical note id for each sample
            result_columns: Columns containing each results 
        """
        combinations = list(itertools.permutations(result_columns, r=2))

        prompts = []
        for combination in combinations:
            for i, row in results.iterrows():
                clinical_note = row[clinical_note_column]
                clinical_note_id = row[clinical_note_id_column]

                if type(row[combination[0]]) == float or type(row[combination[1]]) == float:
                    continue

                a_results = ast.literal_eval(row[combination[0]])[0]
                b_results = ast.literal_eval(row[combination[1]])[0]

                assert len(a_results) == len(b_results), f'The number of retrieved concepts between A {len(a_results)} is different from B ({len(b_results)})'
                for (a_concept, a_result), (b_concept, b_result) in zip(a_results.items(), b_results.items()):
                    assert a_concept == b_concept, f'Found different concepts between the results {a_concept} vs {b_concept} at row {clinical_note_id}'
                    concept_label = self.snomed.get_label_from_id(a_concept)
                    prompt = self.get_prometheus_prompt(clinical_note, concept_label, a_result, b_result, rubric)
                    prompts.append((clinical_note_id, combination[0], combination[1], clinical_note, concept_label, a_result, b_result, prompt))
        prompts = pd.DataFrame(prompts, columns=[
            clinical_note_id_column,
            'a',
            'b',
            clinical_note_column,
            'concept',
            'a_result',
            'b_result',
            'prompt'])
        
        prompts.to_csv(output_file_path)
        return prompts
