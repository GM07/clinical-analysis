from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import joblib

from src.ontology.ontology_filter import BranchesFilter
from src.ontology.snomed import Snomed
from src.ontology.annotator import AnnotationMatch, Annotator

MAX_ANCESTORS = 2

class DomainClassFrequency:
    """
    Class containing the class frequencies for a domain
    """

    EXCLUDE_IDS = ['362981000', '444677008', '419891008', '276339004', '106237007', '900000000000441003']

    def __init__(self, domain: str, frequencies: Dict[str, float]) -> None:
        self.domain = domain
        self.counter = Counter(frequencies)

    def prune_concepts(self, limit: int = 1000):
        """
        Prunes the concepts that are not in the top limit
        """
        self.counter = Counter(dict(self.counter.most_common(limit)))

    def get_concepts(self, top_n: int = None, separate: bool = False):
        """
        Retrieves the concepts in the domain. If top_n is provided, only the top_n concepts are retrieved

        Args:
            top_n: The number of concepts to retrieve
            separate: Whether to return the concepts and frequencies separately
        """
        if separate:
            return self.format_concept_list(self.counter.most_common(top_n))
        else:
            return self.counter.most_common(top_n)

    def format_concept_list(self, concepts: List[Tuple[str, float]]):
        """
        """
        concept_ids = []
        frequencies = []
        for concept in concepts:
            concept_ids.append(concept[0])
            frequencies.append(concept[1])

        return concept_ids, frequencies

    def _hash_to_color(self, text: str) -> str:
        """Convert text to a color using hash"""
        # Use RGB colors with 256 color mode for more variety
        # Each channel (r,g,b) can be 0-5, giving 216 colors
        hash_val = hash(text)
        r = (hash_val & 0xFF) % 6
        g = ((hash_val >> 8) & 0xFF) % 6  
        b = ((hash_val >> 16) & 0xFF) % 6
        
        # Convert to xterm-256 color code (16 + 36*r + 6*g + b)
        color_code = 16 + (36 * r) + (6 * g) + b
        
        return f"\033[38;5;{color_code}m{text}\033[0m"

    def print_most_frequent_concepts(self, snomed: Snomed, top_n: int = 5):
        concepts, frequencies = self.get_concepts(top_n=top_n, separate=True)
        print(f"\n{self.domain}")
        labels = snomed.convert_ids_to_labels(concepts)
        print('\n', '='*100)
        
        for label, freq in zip(labels, frequencies):
            print(f'{label} : {freq}')
        
        print('\n', '='*100)

    @staticmethod
    def get_frequencies_of_domain(domain: str, domain_texts: List[str], snomed: Snomed, annotator: Annotator, concept_limit: int = 1000) -> Dict[str, float]:
        """
        Retrieves the frequencies of each concept in the domain

        Args:
            domain: The domain to analyze
            domain_texts: The texts to analyze
            snomed: The SNOMED ontology
            annotator: The annotator to use
            concept_limit: The maximum number of concepts to analyze
        """
        batches = annotator.batch_annotate(domain_texts, return_ids_only=True)
        concepts = []
        for batch in batches:
            concepts.extend(DomainClassFrequency._get_all_ancestors(batch, snomed))

        filter = BranchesFilter(snomed, DomainClassFrequency.EXCLUDE_IDS)
        concepts = filter(concepts)
        concepts = dict(Counter(concepts).most_common(concept_limit))

        concepts = DomainClassFrequency._get_adjusted_frequencies(Counter(concepts), snomed)
        return DomainClassFrequency(domain, concepts)

    @staticmethod
    def _get_adjusted_frequencies(frequencies: Dict[str, float], snomed: Snomed):
        """
        Adjusts frequencies to favor more general concepts first (generic term will probably
        have less ancestors). Modifies the score according to the following formula :
        new_score = 0.75 * old_score - nb_ancestors * 0.25
        """
        adjusted_frequencies = dict()
        for elem in frequencies.items():
            id, frequency = elem

            # We want to favor more general concepts first 
            # A generic term will probably have less ancestors
            ancestors = snomed.get_ancestors_of_id(id, return_list=True)
            nb_ancestors = max(1, len(ancestors))
            new_frequency = (0.75 * frequency - nb_ancestors * 0.25)
            adjusted_frequencies[id] = new_frequency
        return adjusted_frequencies

    @staticmethod
    def _get_all_ancestors(concept_ids: List[str], snomed: Snomed):
        """
        Retrieves all ancestors of the concept ids present in the list and returns 
        a list containing the ancestors of each concept id and the concept id itself

        Args:
            concept_ids: The concept ids to analyze
            snomed: The SNOMED ontology
        """
        all_concepts = []
        for concept_id in concept_ids:
            all_concepts.extend(snomed.get_ancestors_of_id(concept_id, return_list=True))
        return all_concepts

    @staticmethod
    def get_most_frequent_concepts(text: str, snomed: Snomed, annotator: Annotator, top_n: int = None):
        """
        Retrieves the most frequent concepts of a text

        Args:
            text: The text to analyze
            snomed: The SNOMED ontology
            annotator: The annotator to use
            top_n: The number of concepts to retrieve
        """
        concepts = annotator.annotate(text, return_ids_only=True)
        concepts = DomainClassFrequency._get_all_ancestors(concepts, snomed)

        filter = BranchesFilter(snomed, DomainClassFrequency.EXCLUDE_IDS)
        concepts = filter(concepts)
        
        concepts = DomainClassFrequency._get_adjusted_frequencies(Counter(concepts), snomed)
        return DomainClassFrequency.format_concept_list(concepts.most_common(top_n))
