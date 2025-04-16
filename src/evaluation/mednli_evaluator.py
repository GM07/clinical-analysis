from typing import Callable
import logging

from datasets import Dataset as HuggingFaceDataset

from src.models.halloumi import HallOumi
from src.models.prometheus import Prometheus
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline
from src.pipelines.model_inference_pipeline import ClassifierModelInferencePipeline

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

            prompt = Prometheus.create_prompt(
                instruction=instruction,
                response_a='Yes',
                response_b='No'
            )

            return {'text': prompt}
        
        return dataset.map(apply_template)


class HallOumiMedNLIEvaluator(MedNLIEvaluator):


    def __init__(self, data_path: str, model_path: str, tokenizer_path: str = None, classifier: bool = True, ) -> None:
        self.classifier = classifier
        super().__init__(data_path, model_path, tokenizer_path)

    def load(self):
        self.dataset = HuggingFaceDataset.from_json(self.data_path)

        if self.classifier:
            self.pipeline = ClassifierModelInferencePipeline(self.model_path, self.tokenizer_path)
        else:
            self.pipeline = ModelDatasetInferencePipeline(self.model_path, self.tokenizer_path)

    def __call__(self) -> HuggingFaceDataset:
        """
        Args:
            positive: Whether the LLM answer will be that the statement is factual or not factual (only applicable if `classifier` = False)
        """
        self.dataset = self.preprocess(self.dataset)

        if self.classifier:
            results = self.pipeline.run_inference(
                inputs=self.dataset['text'], 
                batch_size=16, 
                max_length=512,
                apply_chat_template=False,
            )
            dataset = self.dataset.add_column('prediction', results)
            return dataset
        else:
            return self.pipeline(self.dataset['text'])

    def preprocess(self, dataset: HuggingFaceDataset):
        dataset = dataset.filter(lambda x: x['gold_label'] != 'neutral')

        def dataset_func_to_prompt(x):
            prompt = self.get_prompt(x)
            label = True if x['gold_label'] == 'contradiction' else False
            return {'text': prompt, 'label': label}

        df = dataset.map(dataset_func_to_prompt)
        return df

    def get_prompt(self, sample) -> str:

        if self.classifier:
            return HallOumi.create_prompt_classifier(sample['sentence1'], sample['sentence2'])
        return HallOumi.create_prompt_generator(sample['sentence1'], sample['sentence2'], answer='The statement is factual')
