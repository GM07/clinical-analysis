from typing import Callable
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline
from datasets import Dataset as HuggingFaceDataset

import logging

logger = logging.getLogger(__name__)

class MedNLIEvaluator:

    def __init__(self, data_path: str, model_path: str, tokenizer_path: str = None) -> None:
        """
        Args:
            data_path: Path to .json test set of mednli
            model_path: Path to model to evaluate
            tokenizer_path: Path to tokenizer model (optional)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.load()

    def load(self):

        self.dataset = HuggingFaceDataset.from_json(self.data_path)
        self.pipeline = ModelDatasetInferencePipeline(self.model_path, self.tokenizer_path)

    def __call__(
        self, 
        preprocess: Callable[[HuggingFaceDataset], HuggingFaceDataset] = None, 
        input_column: str = 'text',
        apply_chat_template: bool = True,
        system_prompt: str = None,
        max_new_tokens: int = 256
    ) -> HuggingFaceDataset:
        """
        Args:
            preprocess: Preprocess function that modifies the function in order to generate the input column
            input_column: Input that will be sent to the pipeline (default: text)
            apply_chat_template: Whether to apply the chat template before generating (see ModelDatasetInferencePipeline)
            system_prompt: The system prompt to use (only applied if apply_chat_template=True)
        """

        if preprocess:
            self.dataset = preprocess(self.dataset)
        
        print('Sample : ')
        print(self.dataset[0][input_column])

        self.dataset = self.pipeline(
            self.dataset, 
            input_column=input_column, 
            apply_chat_template=apply_chat_template, 
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens
        )
        return self.dataset


class MedNLIPrometheusEvaluator(MedNLIEvaluator):

    PROMPT_TEMPLATE = """###Task Description:
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
[Are the model's responses factually correct and well-supported by evidence?]
Score 1: The model's responses are mostly incorrect or based on unfounded information.
Score 2: The model sometimes provides factually correct responses, but inaccuracies are common.
Score 3: The model generally provides factually correct information, though some errors occur.
Score 4: The model often provides factually accurate information with only occasional minor errors.
Score 5: The model consistently provides responses that are factually correct and well-supported by evidence.

###Feedback: """

    INSTRUCTION_TEMPLATE = """Premise: {premise}
Hypothesis: {hypothesis}

Does this premise entail this hypothesis ? Answer with yes or no only."""

    def __init__(self, data_path: str, model_path: str, tokenizer_path: str = None) -> None:
        super().__init__(data_path, model_path, tokenizer_path)

    def __call__(
        self, 
        input_column: str = 'text', 
        apply_chat_template: bool = True, 
        system_prompt: str = None
    ) -> HuggingFaceDataset:
        return super().__call__(self.preprocess, input_column, apply_chat_template, system_prompt, max_new_tokens=512)
    
    def preprocess(self, dataset):

        dataset = dataset.filter(lambda x: x['gold_label'] != 'neutral')

        def apply_template(x):
            instruction = self.INSTRUCTION_TEMPLATE.format(
                premise=x['sentence1'],
                hypothesis=x['sentence2']
            )

            prompt = self.PROMPT_TEMPLATE.format(
                instruction=instruction,
                response_A='Yes',
                response_B='No'
            )
            return {'text': prompt}
        
        return dataset.map(apply_template)
