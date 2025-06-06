


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
    
    def prune(self, concepts: Dict[str, str], alpha: int = 2, smart: bool = False):
        """
        Returns a new dictionary containing only the valid concepts.

        An concept is valid if it is present in the domain class frequency or if it is within the alpha-th ancestor of a concept 
        that is present in the domain class frequency.

        Args:
            concepts: List of concepts
            alpha: The number of ancestors to consider
            smart: Whether smart pruning is used (only deepest classes and instance will be kept)
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

        filtered = {k: v for k, v in concepts.items() if k in valid_concepts}

        if not smart:
            return filtered

        final_concepts = {k: v for k, v in filtered.items()}
        for k, v in filtered.items():
            nb_children = len(self.snomed.get_children_of_id(k, ids_only=True))
            if nb_children == 0:
                # We keep all instances
                continue
            
            ancestors = self.snomed.get_ancestors_of_id(k, return_set=True)
            if k in ancestors:
                ancestors.remove(k)
            for ancestor in ancestors:
                if ancestor in final_concepts and v == final_concepts[ancestor]:
                    del final_concepts[ancestor]

        return final_concepts

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

    def prune_dataset(self, dataset: ExtractionDataset, input_columns: List[str], alpha: int = 2, smart: bool = False) -> PrunedConceptDataset:
        """
        Prunes the dataset

        Args:
            dataset: The dataset to prune
            input_columns: The columns containing the extractions
            output_column: The column to save the pruned extractions
            alpha: The number of ancestors to consider
            smart: Whether smart pruning is used (only deepest classes and instance will be kept)
        """
        logger.info(f'Pruning with smart = {smart}')
        output_columns = []

        for input_column in input_columns:
            output_column = f'{input_column}_{self.domain_formatted}'
            output_columns.append(output_column)
            logger.info(f'Pruning the dataset with column {input_column} and output column {output_column}')
            dataset.data[output_column] = dataset.data[input_column].apply(lambda x: self.prune(x, alpha=alpha, smart=smart))

        return PrunedConceptDataset(columns=output_columns, data=dataset.data)
