


from typing import Dict, List

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.ontology.snomed import Snomed


class Pruner:
    """
    Takes as an input an extraction (a dictionary of concepts and their answers) and prunes the concepts that are not 
    present in the domain class frequency
    """

    def __init__(self, dcf: DomainClassFrequency, snomed: Snomed):
        self.dcf = dcf
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
                ancestors = set(self.snomed.get_ancestors_of_id(concept, return_list=True)[:alpha])
                intersection = ancestors.intersection(self.allowed_concepts)
                if len(intersection) > 0:
                    valid_concepts.add(concept)

        return {k: v for k, v in concepts.items() if k in valid_concepts}

    def prune_extractions(self, extractions: List[Dict[str, str]]):
        """
        Iterates over the extractions and prunes the concepts in the extractions that are not present in the domain analysis
        """
        pruned_extractions = []
        for extraction in extractions:
            pruned_extraction = self.prune(extraction)
            pruned_extractions.append(pruned_extraction)
        return pruned_extractions

