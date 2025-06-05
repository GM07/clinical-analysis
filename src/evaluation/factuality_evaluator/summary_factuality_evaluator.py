
from typing import Any, Dict
from datasets import Dataset

from src.evaluation.factuality_evaluator.factuality_evaluator import FactualityEvaluator

class SummaryFactualityEvaluator(FactualityEvaluator):

    def __call__(self, dataset: Dataset, out_path: str | None = None, text_column: str = 'TEXT', summary_column: str = 'SUMMARY', id_column: str = 'ROW_ID', apply_chat_template: bool = False, one_shot: Dict[str, Any] = None):
        """
        Evaluates the factual accuracy of the model on a given dataset of summaries.

        Args:
            dataset : The dataset to evaluate (must have the following columns: 'id', 'context', '.
            out_path : The path to the output file to save the results.
            text_column : The name of the column containing the text of the summaries.
            summary_column : The name of the column containing the summary of the text.
            apply_chat_template: Whether to apply the chat template or not
            one_shot: One shot example to add 
        """

        dataset_formatted = dataset.map(lambda x: {
            'id': x[id_column],
            'context': x[text_column],
            'statement': x[summary_column],
        }, remove_columns=[text_column, summary_column, id_column])

        return self.evaluate(dataset_formatted, out_path, apply_chat_template=apply_chat_template, one_shot=one_shot)
