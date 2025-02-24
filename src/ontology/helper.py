


from typing import Dict, List
from src.ontology.annotator import MedCatAnnotator
from src.ontology.snomed import Snomed


class OntologyHelper:


    @staticmethod
    def load_ontology_and_medcat_annotator(snomed_path: str, snomed_cache_path: str, medcat_path: str, medcat_device: str = 'cuda'):
        """
        Loads the ontology and the medcat annotator

        Args:
            snomed_path: Path to the snomed owl file
            snomed_cache_path: Path to the snomed cache file
            medcat_path: Path to the medcat annotator model
            medcat_device: Device used by the medcat annotator
        """
        snomed = Snomed(snomed_path, snomed_cache_path, nb_classes=366771)
        medcat = MedCatAnnotator(medcat_path, device=medcat_device)
        return snomed, medcat


    @staticmethod
    def extracted_ids_to_labels(extracted_ids: List[Dict[str, str]], snomed: Snomed):
        """
        Converts a list of extracted ids to a list of labels.

        Args:
            extracted_ids: List of extracted ids for each clinical note. A single element of the list is a dictionary 
            with the keys being the ids and the values being the extractions for that clinical note.
            snomed: Snomed ontology
        """
        clinical_notes_extractions = []
        for extracted_id in extracted_ids:
            concept_extractions = {}
            for id, extraction in extracted_id.items():
                concept_extractions[snomed.get_label_from_id(id)] = extraction
            clinical_notes_extractions.append(concept_extractions)
        return clinical_notes_extractions
