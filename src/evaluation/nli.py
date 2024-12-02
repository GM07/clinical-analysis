from typing import List
from tqdm import tqdm

import numpy as np
from transformers import pipeline


class NLIEvaluator:
    """
    Evaluates extractions using NLI on two attributes : factuality and relevance
    """

    def __init__(self, nli_model_path: str):
        """
        Args:
            nli_model_path: Path to nli model
        """
        self.nli_model_path = nli_model_path

        self.load()

    def load(self):
        """
        Loads the model
        """
        self.nli_model = pipeline(
            task='zero-shot-classification',
            model=self.nli_model_path,
            local_files_only=True,
            device='cuda'
        )

    def evaluate_factuality(self, clinical_notes: List[str], extractions: List[List[str]], batch_size: int = 8):
        """
        Evaluates the factuality of extractions based on clinical notes. For each pair of
        (clinical_note, extraction), the clinical note is used as the premise and the 
        extraction is used as the hypothesis. For a sample `i`, the note `clinical_note[i]` will
        be used as the premise for the extractions at `extractions[i]`

        Args:
            clinical_notes: List of clinical notes associated to each sample
            extractions: List of all attributes extracted for all clinical notes
            batch_size: Number of samples to process in parallel (real batch will be min(len(extractions_per_note), batch_size))
        """
        assert len(clinical_notes) == len(extractions), f'The number of samples for the extractions {len(extractions)} \
            should be the same as the one for the clinical notes {len(clinical_notes)}'

        factuality_score = 0
        current_length = 0
        for clinical_note, extractions_per_note in tqdm(zip(clinical_notes, extractions), total=len(clinical_notes), desc='Running NLI inference'):
            if len(extractions_per_note) == 0:
                continue
        
            # We cannot batch for more than the number of extractions per note
            scores = self.nli_model(clinical_note, extractions_per_note, multi_label=True, batch_size=min(len(extractions_per_note), batch_size))
            factuality_score += np.mean(scores['scores']) # Average over number of extractions
            current_length += 1

            if current_length % 200 == 0:
                print('current score : ', factuality_score / current_length)

        factuality_score /= min(1, len(clinical_notes))
        return factuality_score
