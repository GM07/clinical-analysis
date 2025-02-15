import logging
import re
from typing import List

import pandas as pd

from src.data.filter import ClinicalAdmissionFilter

logger = logging.getLogger(__name__)

class Mimic:
    """
    Loads the MIMIC dataset from the noteevents.csv file.

    This class will : 
    - Load the dataset in a pandas DataFrame
    - Regroup all notes from the same admission 
    - Add a column called `note_id` on the dataset which indicates the order of the note in the admission
    """

    def __init__(self, mimic_path: str, load: bool = True):
        """
        Initializes a new loader

        Args:
            mimic_path: Path to the noteevents.csv file of the MIMIC dataset
            remove_admissions_without_summary: Removes all admissions that do not contain a discharge summary in their set of notes
            load_and_format: Whether to directly load and the data or not
        """
        self.mimic_path = mimic_path

        self.data = None
        self.formatted_data = None

        if load:
            self.load()

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset in a pandas DataFrame
        """
        if self.data is not None:
            return self.data

        self.data = pd.read_csv(self.mimic_path)
        return self.data

    def format(self, remove_admissions_without_summary: bool = True) -> pd.DataFrame:
        """
        Regroups all notes from the same admission, sorts notes of the same admission based on the date and time.

        Args:
            remove_admissions_without_summary: Remove all admission notes that do not contain a discharge summary
        """
        assert self.data is not None, 'The load() function must be called prior to the format() function'

        # Remove rows with N/A in the columns TEXT, 'HADM_ID', 'CHARTDATE', 'CATEGORY' or 'DESCRIPTION'
        formatted_data = self.data[self.data['TEXT'].notna()]
        formatted_data = formatted_data[formatted_data['HADM_ID'].notna()]
        formatted_data = formatted_data[formatted_data['CHARTDATE'].notna()]
        formatted_data = formatted_data[formatted_data['CATEGORY'].notna()]
        formatted_data = formatted_data[formatted_data['DESCRIPTION'].notna()]

        # Sort according to HADM_ID, CHARTDATE and CHARTTIME
        formatted_data = formatted_data.sort_values(by=['HADM_ID', 'CHARTDATE', 'CHARTTIME'])

        # Remove notes without a discharge summary (notes with more than 1 discharge summary are also removed)
        if remove_admissions_without_summary:

            # Count discharge summaries per admission
            discharge_counts = formatted_data[formatted_data['CATEGORY'] == 'Discharge summary']['HADM_ID'].value_counts()
            
            # Get admissions with exactly one discharge summary
            single_discharge_admissions = discharge_counts[discharge_counts == 1].index
            
            # Filter the original dataframe
            formatted_data = formatted_data[formatted_data['HADM_ID'].isin(single_discharge_admissions)]
            
        else:
            formatted_data = formatted_data
        
        formatted_data['NOTE_ORDER'] = formatted_data.groupby('HADM_ID').cumcount()
        self.formatted_data = formatted_data.copy()
        return self.formatted_data

    def filter(self, filter: ClinicalAdmissionFilter) -> pd.DataFrame:
        return self.formatted_data.groupby('HADM_ID').filter(lambda x: filter(x['TEXT'].tolist()))


    def get_excluded_notes(self, excluded_note_ids: List[int]) -> List[str]:
        """
        Returns the notes that are not in the excluded_hadm_ids list

        Args:
            excluded_hadm_ids: List of admission ids to exclude
        """
        return self.data[~self.data['ROW_ID'].isin(excluded_note_ids)]


    @staticmethod
    def get_note_ids_from_path(path: str) -> List[int]:
        """
        Returns the note ids present in the processed file

        Args:
            path: Path to the processed file
        """
        processed_data = pd.read_csv(path)
        return processed_data['ROW_ID'].tolist()


class BHCExtractor:
    """
    Extracts the BHC from the MIMIC dataset
    """

    BHC_REGEX = r'(history of present illness|history|hospital course|present illness):\s*([^\n]*(?:\n(?!\n)[^\n]*)*)'

    def __init__(self, data: pd.DataFrame = None, processed_mimic_path: str = None):

        assert data is not None or processed_mimic_path is not None, 'Either data or processed_mimic_path must be provided'
        assert data is None or processed_mimic_path is None, 'Only one of data or processed_mimic_path must be provided'

        if data is not None:
            self.processed_data = data
        else:
            self.processed_data = pd.read_csv(processed_mimic_path)

    def extract(self) -> pd.DataFrame:
        """
        Extracts the BHC from the MIMIC dataset by only adding a value to the 'BHC' column for the discharge summaries
        """

        # Get the discharge summaries
        discharge_summaries = self.processed_data['CATEGORY'] == 'Discharge summary'

        # Create a new column called 'BHC'
        self.processed_data.loc[:, 'BHC'] = None

        def extract_bhc(text: str) -> str:
            match = re.search(self.BHC_REGEX, text, re.IGNORECASE)
            if match:
                return match.group(2)
            return None

        self.processed_data.loc[discharge_summaries, 'BHC'] = self.processed_data.loc[discharge_summaries, 'TEXT'].apply(extract_bhc)

        return self.processed_data
