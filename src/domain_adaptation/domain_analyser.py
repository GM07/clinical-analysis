import logging
import joblib
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from src.data.mimic import Mimic
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed


logger = logging.getLogger(__name__)

class DomainAnalyser:
    """
    Processes the Mimic data and then generates the DCF of each domain
    """

    def __init__(self, mimic_path: str, processed_path: str = None):
        self.mimic_path = mimic_path
        self.processed_path = processed_path

        self.load_mimic()

        self.domains = self.data['CATEGORY'].unique()
        self.domain_class_frequencies = {}

    def load_mimic(self):
        self._mimic_loader = Mimic(self.mimic_path)
        if self.processed_path is None:
            processed_ids = self._mimic_loader.format()['ROW_ID'].tolist()
        else:
            processed_ids = self._mimic_loader.get_note_ids_from_path(self.processed_path)

        self.data = self._mimic_loader.get_excluded_notes(processed_ids)
        

    def cap_domains(self, limit: int = 1000):
        """
        Will filter the data to include all domains up to the limit

        Args:
            limit: The maximum number of notes to include
        """
        for domain in self.domains:
            domain_data = self.data[self.data['CATEGORY'] == domain].head(limit)
            if domain == self.domains[0]:
                self.domain_data = domain_data
            else:
                self.domain_data = pd.concat([self.domain_data, domain_data])

        return self.domain_data

    def generate_domain_class_frequencies(self, snomed: Snomed, annotator: Annotator, limit: int = 1000, concept_limit: int = 1000):
        """
        Generates the class frequencies for each domain

        Args:
            snomed: The SNOMED ontology
            annotator: An annotator that returns snomed concepts
            limit: The maximum number of notes to include
        """
        self.domain_class_frequencies = {}
        self.cap_domains(limit)
        for domain in tqdm(self.domains, desc='Generating domain class frequencies'):
            domain_data = self.domain_data[self.domain_data['CATEGORY'] == domain]
            domain_class_frequency = DomainClassFrequency.get_frequencies_of_domain(domain, domain_data['TEXT'].tolist(), snomed, annotator, concept_limit)
            self.domain_class_frequencies[domain] = domain_class_frequency
        self.normalize_domain_class_frequencies(limit=limit)
        return self.domain_class_frequencies

    def compute_average_concept_frequencies(self):
        """
        Computes the average frequency of each concept across all domains for a single note
        """
        average_concept_frequencies = defaultdict(int)
        for domain in self.domain_class_frequencies:
            for concept, frequency in self.domain_class_frequencies[domain].counter.items():
                average_concept_frequencies[concept] += frequency / len(self.domains)
        return average_concept_frequencies

    def normalize_domain_class_frequencies(self):
        """
        Normalizes the domain class frequencies to find the average frequency of each concept across all domains for a single note
        """
        
        average_concept_frequencies = self.compute_average_concept_frequencies()

        logger.info(f'Updating frequencies for {len(self.domains)} domains')
        for domain in self.domain_class_frequencies:
            for concept, frequency in self.domain_class_frequencies[domain].frequencies.items():
                self.domain_class_frequencies[domain].frequencies[concept] = frequency - average_concept_frequencies[concept]

            self.domain_class_frequencies[domain].counter = Counter(self.domain_class_frequencies[domain].frequencies)

        return self.domain_class_frequencies

    def prune_concepts(self, limit: int = 1000):
        """
        Prunes the concepts that are not in the top limit
        """
        for domain in self.domain_class_frequencies:
            self.domain_class_frequencies[domain].prune_concepts(limit)

        return self.domain_class_frequencies

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
