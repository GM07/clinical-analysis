


import logging
from typing import Dict, List

from src.data.dataset import ExtractionDataset, PrunedConceptDataset
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)
class Pruner:
    """
    Takes as an input an extraction (a dictionary of concepts and their answers) and prunes the concepts that are not 
    present in the domain class frequency
    """

    def __init__(self, dcf: DomainClassFrequency, snomed: Snomed):
        self.dcf = dcf
        self.domain = dcf.domain
        self.domain_formatted = self.domain.replace(' ', '_').replace('/', '_').lower()
        self.snomed = snomed
        self.allowed_concepts = set(dcf.counter.keys())
    
    def prune(self, concepts: Dict[str, str], alpha: int = 2):
        """
        Returns a new dictionary containing only the valid concepts.

        An concept is valid if it is present in the domain class frequency or if it is within the alpha-th ancestor of a concept 
        that is present in the domain class frequency.
        """
        valid_concepts = set() 
        for concept in concepts:
            if concept in self.allowed_concepts:
                valid_concepts.add(concept)
            else:
                ancestors = self.snomed.get_ancestors_of_id(concept, return_list=True)
                ancestors = set(ancestors[:min(alpha, len(ancestors))])
                intersection = ancestors.intersection(self.allowed_concepts)
                if len(intersection) > 0:
                    valid_concepts.add(concept)

        return {k: v for k, v in concepts.items() if k in valid_concepts}

    def prune_extractions(self, extractions: List[Dict[str, str]], alpha: int = 2):
        """
        Iterates over the extractions and prunes the concepts in the extractions that are not present in the domain analysis

        Args:
            extractions: List of extractions
            alpha: The number of ancestors to consider
        """
        pruned_extractions = []
        for extraction in extractions:
            pruned_extraction = self.prune(extraction, alpha)
            pruned_extractions.append(pruned_extraction)
        return pruned_extractions

    def prune_dataset(self, dataset: ExtractionDataset, input_column: str, alpha: int = 2) -> PrunedConceptDataset:
        """
        Prunes the dataset

        Args:
            dataset: The dataset to prune
            input_column: The column containing the extractions
            output_column: The column to save the pruned extractions
            alpha: The number of ancestors to consider
        """
        output_column = f'{input_column}_{self.domain_formatted}'
        logger.info(f'Pruning the dataset with column {input_column} and output column {output_column}')
        dataset.data[output_column] = dataset.data[input_column].apply(lambda x: self.prune(x, alpha=alpha))
        return PrunedConceptDataset(columns=[output_column], data=dataset.data)
