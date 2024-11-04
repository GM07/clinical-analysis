import logging

from transformers import AutoTokenizer

from src.loader import MimicLoader
from src.filter import ComposedFilter, NoteCountFilter, TokenLengthFilter
from src.pipelines.pipeline import Pipeline


logger = logging.getLogger(__name__)

class BhcTaskDatasetPipeline(Pipeline):
    """
    Generates the dataset for the BHC task using the MIMIC-III dataset
    """

    def __init__(
        self, 
        mimic_path: str, 
        output_file_path: str,
        tokenizer_path: str, 
        max_nb_notes: int = 10, 
        max_tokens_per_note: int = 2048
    ):
        super().__init__(output_file_path)
        self.mimic_path = mimic_path
        self.tokenizer_path = tokenizer_path
        self.max_nb_notes = max_nb_notes
        self.max_tokens_per_note = max_tokens_per_note
        logger.info(f'[INIT] Pipeline initialised with self.max_nb_notes={self.max_nb_notes}, self.max_tokens_per_note={self.max_tokens_per_note}')


    def __call__(self):

        loader = MimicLoader(self.mimic_path)
        logger.info(f'Number of admissions when loaded : {len(loader.data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes when loaded : {len(loader.data)}')

        formatted_data = loader.format(remove_admissions_without_summary=True)

        logger.info(f'Number of admissions after formatting : {len(formatted_data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes after formatting : {len(formatted_data)}')

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        filter = ComposedFilter([NoteCountFilter(self.max_nb_notes), TokenLengthFilter(tokenizer, self.max_tokens_per_note)])
        filtered_data = loader.filter(filter)

        logger.info(f'Number of admissions after filtering : {len(filtered_data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes after filtering : {len(filtered_data)}')

        filtered_data.to_csv(self.output_file_path)

