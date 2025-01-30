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
        self.frequencies = frequencies
        self.counter = Counter(frequencies)

    def get_concepts(self, top_n: int = None):
        """
        Retrieves the concepts in the domain. If top_n is provided, only the top_n concepts are retrieved

        Args:
            top_n: The number of concepts to retrieve
        """
        return self.format_concept_list(self.counter.most_common(top_n))

    def format_concept_list(self, concepts: List[Tuple[str, float]]):
        """
        """
        concept_ids = []
        frequencies = []
        for concept in concepts:
            concept_ids.append(concept[0])
            frequencies.append(concept[1])

        return concept_ids, frequencies

    @staticmethod
    def get_frequencies_of_domain(domain: str, domain_texts: List[str], snomed: Snomed, annotator: Annotator) -> Dict[str, float]:
        """
        Retrieves the frequencies of each concept in the domain

        Args:
            domain: The domain to analyze
            domain_texts: The texts to analyze
            snomed: The SNOMED ontology
            annotator: The annotator to use
        """
        batches = annotator.batch_annotate(domain_texts, return_ids_only=True)
        concepts = []
        for batch in batches:
            concepts.extend(DomainClassFrequency._get_all_ancestors(batch, snomed))

        filter = BranchesFilter(snomed, DomainClassFrequency.EXCLUDE_IDS)
        concepts = filter(concepts)

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
