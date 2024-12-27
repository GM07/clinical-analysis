from typing import List
from abc import ABC

from transformers import AutoTokenizer

class ClinicalAdmissionFilter(ABC):
    # TODO : Change this to Validator

    def __call__(self, notes: List[str]) -> bool:
        """
        Returns whether a set of notes linked to a clinical admission is valid or not based on a certain criteria.

        Args:
            notes: List of notes linked to the admission

        Returns
        Whether the admission is valid or not
        """
        pass

class ComposedFilter(ClinicalAdmissionFilter):
    """
    Invalidates a clinical admission based on multiple filters
    """

    def __init__(self, filters: List[ClinicalAdmissionFilter]):
        super().__init__()
        self.filters = filters

    def __call__(self, notes: List[str]) -> bool:
        for filter in self.filters:
            valid = filter(notes)
            if not valid:
                return False
            
        return True

class TokenLengthFilter(ClinicalAdmissionFilter):
    """
    Invalidates a clinical admission if one of the notes has a certain number of tokens.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_token_length: int):
        """
        Args:
            tokenizer: Tokenizer to use when tokenizing the clinical notes
            max_token_length: Number of maximal tokens in a clinical note (valid if equal)
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __call__(self, notes: List[str]) -> bool:
        tokenized_notes = self.tokenizer(notes)['input_ids']
        for tokenized_note in tokenized_notes:
            if len(tokenized_note) > self.max_token_length:
                return False

        return True


class NoteCountFilter(ClinicalAdmissionFilter):
    """
    Invalidates a clinical admission if it contains more than a certain number of notes
    """

    def __init__(self, max_nb_notes: int):
        """
        Args:
            max_nb_notes: Max number of clinical notes allowed in an admission (valid if equal)
        """
        super().__init__()

        self.max_nb_notes = max_nb_notes

    def __call__(self, notes: List[str]) -> bool:
        return len(notes) <= self.max_nb_notes
