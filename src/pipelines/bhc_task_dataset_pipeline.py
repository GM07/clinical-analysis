import logging

from transformers import AutoTokenizer

from src.data.mimic import Mimic, BHCExtractor
from src.data.filter import ComposedFilter, NoteCountFilter, TokenLengthFilter
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
        max_nb_notes_per_adm: int = 10, 
        max_tokens_per_note: int = 2048,
        max_total_nb_notes: int = None,
        extract_bhc: bool = True
    ):
        """
        Args:
            mimic_path (str): Path to the MIMIC-III dataset
            output_file_path (str): Path to save the output file
            tokenizer_path (str): Path to the tokenizer
            max_nb_notes_per_adm (int): Maximum number of notes per admission
            max_tokens_per_note (int): Maximum number of tokens per note
            max_total_nb_notes (int): Maximum number of notes to include
        """

        self.output_file_path = output_file_path
        self.mimic_path = mimic_path
        self.tokenizer_path = tokenizer_path
        self.max_nb_notes_per_adm = max_nb_notes_per_adm
        self.max_tokens_per_note = max_tokens_per_note
        self.max_total_nb_notes = max_total_nb_notes
        self.extract_bhc = extract_bhc
        logger.info(f'[INIT] Pipeline initialised with self.max_nb_notes_per_adm={self.max_nb_notes_per_adm}, self.max_tokens_per_note={self.max_tokens_per_note}')


    def __call__(self):

        loader = Mimic(self.mimic_path)
        logger.info(f'Number of admissions when loaded : {len(loader.data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes when loaded : {len(loader.data)}')

        formatted_data = loader.format(remove_admissions_without_summary=True)

        logger.info(f'Number of admissions after formatting : {len(formatted_data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes after formatting : {len(formatted_data)}')

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        filter = ComposedFilter([NoteCountFilter(self.max_nb_notes_per_adm), TokenLengthFilter(tokenizer, self.max_tokens_per_note)])
        filtered_data = loader.filter(filter)

        logger.info(f'Number of admissions after filtering : {len(filtered_data.HADM_ID.unique())}')
        logger.info(f'Number of clinical notes after filtering : {len(filtered_data)}')

        if self.max_total_nb_notes is not None:
            # Get all rows up to and including the complete last admission
            mask = (filtered_data['HADM_ID'].isin(
                filtered_data.iloc[:self.max_total_nb_notes]['HADM_ID'].unique()
            ))

            capped_data = filtered_data[mask]

            logger.info(f'Number of admissions after capping to {self.max_total_nb_notes} : {len(capped_data.HADM_ID.unique())}')
            logger.info(f'Number of clinical notes after capping to {self.max_total_nb_notes} : {len(capped_data)}')


            if self.extract_bhc:
                extractor = BHCExtractor(data=capped_data)
                capped_data = extractor.extract()

            capped_data.to_csv(self.output_file_path)
            return


        if self.extract_bhc:
            extractor = BHCExtractor(data=filtered_data)
            filtered_data = extractor.extract()

        filtered_data.to_csv(self.output_file_path)

