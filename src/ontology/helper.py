


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
