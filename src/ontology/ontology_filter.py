


from abc import ABC, abstractmethod
from typing import List, Set

from src.ontology.snomed import Snomed


class OntologyFilter(ABC):

    @abstractmethod
    def __call__(self, concept_ids: List[str]):
        """
        Filters ontological concepts
        """
        pass

class ComposedFilter(OntologyFilter):

    def __init__(self, ontology_filters: List[OntologyFilter]):
        super().__init__()
        self.ontology_filters = ontology_filters


    def __call__(self, concept_ids: List[str]):
        for ontology_filter in self.ontology_filters:
            concept_ids = ontology_filter(concept_ids)
        return concept_ids

class BranchFilter(OntologyFilter):
    """
    Filter that will remove all descendants of a base class (thus pruning a whole branch of the ontology)
    """

    def __init__(self, snomed: Snomed, base_concept_id: str):
        super().__init__()

        self.snomed = snomed
        self.base_concept_id = base_concept_id

    def __call__(self, concept_ids: List[str]):
        """
        Filters ontological concepts (keeps the order of the list)
        """
        filtered_concept_ids = []
        for concept_id in concept_ids:
            if concept_id == self.base_concept_id:
                continue

            ancestors: set = self.snomed.get_ancestors_of_id(concept_id, return_set=True)
            if self.base_concept_id not in ancestors and len(ancestors) > 0:
                filtered_concept_ids.append(concept_id)
        return filtered_concept_ids


class BranchesFilter(OntologyFilter):
    """
    Filter that will remove all descendants of multiple base classes (thus pruning branches of the ontology)
    """

    def __init__(self, snomed: Snomed, base_concept_ids: List[str]):
        super().__init__()

        self.snomed = snomed
        self.base_concept_ids = set(base_concept_ids)

    def __call__(self, concept_ids: List[str]):
        """
        Filters ontological concepts (keeps the order of the list)
        """
        filtered_concept_ids = []
        for concept_id in concept_ids:
            if concept_id in self.base_concept_ids:
                continue
            
            ancestors: set = self.snomed.get_ancestors_of_id(concept_id, return_set=True)
            intersect = self.base_concept_ids.intersection(ancestors)
            if len(ancestors) > 0 and len(intersect) == 0:
                filtered_concept_ids.append(concept_id)
        return filtered_concept_ids
