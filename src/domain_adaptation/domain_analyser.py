import logging
import os
from typing import List
import joblib
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from src.data.mimic import BHCExtractor, Mimic
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.ontology.annotator import Annotator
from src.ontology.snomed import Snomed


logger = logging.getLogger(__name__)

class DomainAnalyser:
    """
    Processes the Mimic data and then generates the DCF of each domain
    """

    def __init__(self, mimic_path: str, processed_mimic_path: str = None, domains: List[str] = None):
        """
        Args:
            mimic_path: The path to the Mimic data
            processed_mimic_path: The path to the processed Mimic data
            domains: The domains to process (if None, all detected domains are processed)
        """
        self.mimic_path = mimic_path
        self.processed_mimic_path = processed_mimic_path

        self.load_mimic()

        if domains is None:
            self.domains = self.data['CATEGORY'].unique()
        else:
            self.domains = domains
        self.domain_class_frequencies = {}

    def load_mimic(self):
        self._mimic_loader = Mimic(self.mimic_path)
        if self.processed_mimic_path is None:
            processed_ids = self._mimic_loader.format()['ROW_ID'].tolist()
        else:
            processed_ids = self._mimic_loader.get_note_ids_from_path(self.processed_mimic_path)

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
                domain_data = domain_data
            else:
                domain_data = pd.concat([domain_data, domain_data])

        return domain_data

    def generate_domain_class_frequencies(self, snomed: Snomed, annotator: Annotator, limit: int = 1000, concept_limit: int = 1000):
        """
        Generates the class frequencies for each domain

        Args:
            snomed: The SNOMED ontology
            annotator: An annotator that returns snomed concepts
            limit: The maximum number of notes to include
        """
        self.domain_class_frequencies = {}
        domain_data = self.cap_domains(limit)
        for domain in tqdm(self.domains, desc='Generating domain class frequencies'):
            domain_data = domain_data[domain_data['CATEGORY'] == domain]
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
                average_concept_frequencies[concept] += frequency / len(self.domain_class_frequencies)
        return average_concept_frequencies

    def normalize_domain_class_frequencies(self):
        """
        Normalizes the domain class frequencies to find the average frequency of each concept across all domains for a single note
        """
        
        average_concept_frequencies = self.compute_average_concept_frequencies()

        logger.info(f'Updating frequencies for {len(self.domain_class_frequencies)} domains')
        for domain in self.domain_class_frequencies:
            for concept, frequency in self.domain_class_frequencies[domain].counter.items():
                self.domain_class_frequencies[domain].counter[concept] = frequency - average_concept_frequencies[concept]

            self.domain_class_frequencies[domain].counter = Counter(self.domain_class_frequencies[domain].counter)

        return self.domain_class_frequencies

    def add_domain_class_frequencies(self, domain: str, domain_class_frequency: DomainClassFrequency):
        """
        Adds a domain class frequency to the domain class frequencies

        Args:
            domain: The domain to add the class frequencies to
            domain_class_frequency: The domain class frequencies to add
        """
        self.domain_class_frequencies[domain] = domain_class_frequency

    def prune_concepts(self, concept_limit: int = 1000):
        """
        Prunes the concepts that are not in the top limit

        Args:
            concept_limit: The number of concepts to keep
        """
        for domain in self.domain_class_frequencies:
            self.domain_class_frequencies[domain].prune_concepts(concept_limit)

        return self.domain_class_frequencies

    def save(self, path: str, separate_files: bool = False):
        """
        Saves the domain class frequencies

        Args:
            path: The path to save the domain class frequencies
            separate_files: Whether to save the domain class frequencies in separate files (if true, the path will be a 
            folder and each domain will have its own file under the domain name)
        """
        if separate_files:
            if not os.path.exists(path):
                os.mkdir(path)
            for domain in self.domain_class_frequencies:
                file_name = domain.replace(' ', '_').replace('/', '_').lower()
                self.domain_class_frequencies[domain].save(path + f'/{file_name}.dcf')
        else:
            joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class BHCDomainAnalyser(DomainAnalyser):
    """
    Processes the Mimic data and then generates the DCF of each domain
    """

    def __init__(self, mimic_path: str, processed_mimic_path: str):
        self.mimic_path = mimic_path
        self.processed_mimic_path = processed_mimic_path

        self.load_mimic()
        self.domain_class_frequencies = {}

    def generate_bhc_class_frequencies(self, snomed: Snomed, annotator: Annotator, limit: int = 1000, concept_limit: int = 1000):
        """
        Generates the class frequencies of the BHC section of the discharge summaries
        """
        discharge_summaries = self.data[self.data['CATEGORY'] == 'Discharge summary'].head(limit)
        bhc_data = BHCExtractor(data=discharge_summaries).extract()

        bhc_class_frequency = DomainClassFrequency.get_frequencies_of_domain('BHC', bhc_data['BHC'].tolist(), snomed, annotator, concept_limit)
        self.domain_class_frequencies['BHC'] = bhc_class_frequency

        # discharge_frequency = DomainClassFrequency.get_frequencies_of_domain('Discharge summary', discharge_summaries['TEXT'].tolist(), snomed, annotator, concept_limit)
        # self.domain_class_frequencies['Discharge summary'] = discharge_frequency

        # self.normalize_domain_class_frequencies()
        return self.domain_class_frequencies['BHC']
