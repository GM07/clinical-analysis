import re
from typing import List

from src.data.dataset import VerbalizedExtractionDataset

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from transformers import pipeline


class DomainAdaptationEvaluator:
    """
    Evaluator for domain adaptation
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()

        self.model_labels_to_domains = {
            'Physician ': 'physician_',
            'Radiology': 'radiology',
            'Nursing/other': 'nursing_other',
            'ECG': 'ecg',
        }

        self.domains_to_model_labels = {
            'physician_': 'Physician ',
            'radiology': 'Radiology',
            'nursing_other': 'Nursing/other',
            'ecg': 'ECG',
        }

    def load_model(self):
        model = pipeline("text-classification", model=self.model_path)
        return model

    def evaluate_verbalized_dataset(self, dataset: VerbalizedExtractionDataset):
        dataset.filter_non_valid_generations()

        final_scores = {}
        average = 0
        lengths = {}
        for domain, column in zip(dataset.domains, dataset.columns):
            results = dataset.data[column]
            results = results[results.notna()]
            results = results.tolist()
            lengths[domain] = len(results)
            results = self._remove_domain_mention_in_texts(results)

            final_scores[domain] = self._get_domain_score(results, domain)
            average += final_scores[domain]

        average /= len(lengths)
        final_scores['average'] = average
        return final_scores

    def evaluate_huggingface_dataset(self, dataset: HuggingFaceDataset, input_column: str = 'OUTPUT', domain_column: str = 'CATEGORY'):
        dataset = dataset.filter(lambda x: x[input_column] is not None and x[domain_column] is not None)
        ecg_rows = dataset.filter(lambda x: x[domain_column] == 'ECG')
        radiology_rows = dataset.filter(lambda x: x[domain_column] == 'Radiology')
        nursing_rows = dataset.filter(lambda x: x[domain_column] == 'Nursing')
        physician_rows = dataset.filter(lambda x: x[domain_column] == 'Physician')

        # We don't want to mention the domain in the results as it might bias the model, so we replace every mention of the domain with the word 'clinical'
        # Regex to replace the domain with the word 'clinical'
        ecg_results = self._remove_domain_mention_in_texts(ecg_rows[input_column])
        radiology_results = self._remove_domain_mention_in_texts(radiology_rows[input_column])
        nursing_results = self._remove_domain_mention_in_texts(nursing_rows[input_column])
        physician_results = self._remove_domain_mention_in_texts(physician_rows[input_column])

        ecg_score = self._get_domain_score(ecg_results, 'ecg')
        radiology_score = self._get_domain_score(radiology_results, 'radiology')
        nursing_score = self._get_domain_score(nursing_results, 'nursing_other')
        physician_score = self._get_domain_score(physician_results, 'physician_')

        average = ecg_score + radiology_score + nursing_score + physician_score
        average /= 4

        return {
            'ecg': ecg_score,
            'radiology': radiology_score,
            'nursing_other': nursing_score,
            'physician_': physician_score,
            'average': average
        }

    def _get_domain_score(self, results: List[str], domain: str):
        scores = self.model(results, batch_size=8, top_k=4, max_length=512, truncation=True)
        scores = [{score['label']: score['score'] for score in score_list} for score_list in scores]
        expected_label = self.domains_to_model_labels[domain]
        domain_score = np.array([score[expected_label] for score in scores])
        return domain_score.mean()

    def _remove_domain_mention_in_texts(self, texts: List[str]):
        regex = r"ECG|Radiology|Nursing|Nursing/Other|Physician"
        return list(map(lambda x: re.sub(regex, 'clinical', x, flags=re.IGNORECASE), texts))
