
from datasets import Dataset

from src.data.dataset import ExtractionDataset
from src.evaluation.factuality_evaluator.factuality_evaluator import FactualityEvaluator
from src.ontology.snomed import Snomed


class ExtractionFactualityEvaluator(FactualityEvaluator):


    def __init__(self, model_path: str, snomed_path: str, snomed_cache_path: str, tokenizer_path: str | None = None) -> None:
        super().__init__(model_path, tokenizer_path)
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path
        

    def load(self):
        super().load()
        self.snomed = Snomed(path=self.snomed_path, cache_path=self.snomed_cache_path)

    def __call__(self, extraction_dataset: ExtractionDataset, method: str = 'constrained', out_path: str | None = None, extractions_only: bool = True):
        total_extractions = extraction_dataset._get_extractions(method)
        clinical_notes = extraction_dataset.clinical_notes()
        ids = extraction_dataset.clinical_note_ids()

        dataset = {
            'id': [],
            'context': [],
            'statement': [],
        }
        for id, clinical_note, extractions in zip(ids, clinical_notes, total_extractions):
            for concept, extraction in extractions.items():
                if len(concept) == 0:
                    continue
                label = self.snomed.get_label_from_id(concept)
                statement = extraction if extractions_only else f'{label} : {extraction}'
                dataset['id'].append(id)
                dataset['context'].append(clinical_note)
                dataset['statement'].append(statement)

        dataset = Dataset.from_dict(dataset)
        return self.evaluate(dataset, out_path)
