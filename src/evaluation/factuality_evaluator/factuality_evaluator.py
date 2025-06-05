import logging
from typing import Any, Dict
from datasets import Dataset

from src.data.formatter import Formatter
from src.evaluation.medhal_parser import MedHalParser
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline

logger = logging.getLogger(__name__)

class FactualityEvaluator:

    def __init__(self, model_path: str, tokenizer_path: str | None = None) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.load()

    def load(self):
        self.pipeline = ModelDatasetInferencePipeline(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path
        )

        self.formatter = Formatter(tokenizer=self.pipeline.tokenizer, training=False)
        self.parser = MedHalParser()

    def unload(self):
        del self.pipeline

    def evaluate(self, dataset: Dataset, out_path: str | None = None, apply_chat_template = False, one_shot: Dict[str, Any] = None):
        """
        Parses the result of the model and returns the metrics.

        Args:
            dataset: The dataset to evaluate (must have the following columns: 'id', 'context', 'statement').
            out_path: The path to the output file to save the results.
            apply_chat_template: Whether to apply the chat template or not
            one_shot: One shot example to give in each sample (must have these keys : context, statement, label and explanation)
        """
        filtered = dataset.filter(lambda x: x['statement'] is not None)
        if len(dataset) != len(filtered):
            logger.warning('Some samples could not be evaluated as their statement is None')
            
        if one_shot:
            assert 'context' in one_shot.keys() and 'statement' in one_shot.keys() and 'label' in one_shot.keys() and 'explanation' in one_shot.keys(), 'A context, a statement, a label and an explanation must be provided in the one shot example'
            # formatted_one_shot = self.formatter.format_one_shot(one_shot)
            formatted = filtered.map(lambda x: {'text': self.formatter.format_one_shot_with_sample(
                context=x['context'],
                statement=x['statement'],
                context_one_shot=one_shot['context'],
                statement_one_shot=one_shot['statement'],
                label_one_shot=one_shot['label'],
                explanation_one_shot=one_shot['explanation'],
            )}, load_from_cache_file=False)
        else:
            formatted = filtered.map(self.formatter, load_from_cache_file=False)
        result = self.pipeline(formatted, input_column='text', output_column='output', apply_chat_template=apply_chat_template, max_new_tokens=128)

        if out_path:
            result.to_csv(out_path)

        if one_shot:
            result = result.map(lambda x: {'text': x['text'].replace(self.formatter.format_one_shot(
                context_one_shot=one_shot['context'],
                statement_one_shot=one_shot['statement'],
                label_one_shot=one_shot['label'],
                explanation_one_shot=one_shot['explanation'],
            ), ''), 'prompt': x['text']}, load_from_cache_file=False)

        metrics = self.parser.parse(result, add_prompt=True, output_col='output', prompt_col='text')
        return metrics


