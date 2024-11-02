import pandas as pd
import logging

from src.filter import ClinicalAdmissionFilter


logger = logging.getLogger(__name__)

class MimicLoader:
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
        Regroups all notes from the same admission, sorts notes of the same admission based on the date and time. Removes
        clinical notes that 
        """
        assert self.data is not None, 'The load() function must be called prior to the format() function'

        if self.formatted_data is not None:
            return self.formatted_data
        
        # Remove rows with N/A in the columns TEXT, 'HADM_ID', 'CHARTDATE', 'CATEGORY' or 'DESCRIPTION'
        formatted_data = self.data[self.data['TEXT'].notna()]
        formatted_data = formatted_data[formatted_data['HADM_ID'].notna()]
        formatted_data = formatted_data[formatted_data['CHARTDATE'].notna()]
        formatted_data = formatted_data[formatted_data['CATEGORY'].notna()]
        formatted_data = formatted_data[formatted_data['DESCRIPTION'].notna()]

        # Sort according to HADM_ID, CHARTDATE and CHARTTIME
        formatted_data = formatted_data.sort_values(by=['HADM_ID', 'CHARTDATE', 'CHARTTIME'])

        # Remove notes without a discharge summary
        if remove_admissions_without_summary:
            only_discharge = formatted_data[formatted_data['CATEGORY'] == 'Discharge summary']['HADM_ID'].unique()
            formatted_data = formatted_data[formatted_data['HADM_ID'].isin(only_discharge)]    
        else:
            formatted_data = formatted_data
        
        formatted_data['NOTE_ORDER'] = formatted_data.groupby('HADM_ID').cumcount()
        self.formatted_data = formatted_data.copy()
        return self.formatted_data

    def filter(self, filter: ClinicalAdmissionFilter) -> pd.DataFrame:
        return self.formatted_data.groupby('HADM_ID').filter(lambda x: filter(x['TEXT'].tolist()))