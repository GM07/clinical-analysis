
from typing import Dict, List
from medcat.cat import CAT

from src.ontology.annotator import Annotator, AnnotationMatch

class MedCatAnnotator(Annotator):
    """
    Annotator based on the MedCAT model
    """

    def __init__(self, medcat_path: str, device: str = None, meta_cat_config_dict: Dict = None) -> None:
        """
        Args:
            medcat_path: Path to medcat model
            device: Which device to put the annotator on
            meta_cat_config_dict: Configuration dictionary of the model. More details at https://github.com/CogStack/MedCAT
        """
        self.path = medcat_path
        if device is not None and meta_cat_config_dict is None:
            config={
                'general': {
                    'device': device
                }
            }
        else:
            config = meta_cat_config_dict

        self.cat = CAT.load_model_pack(medcat_path, meta_cat_config_dict=config)
        self.cat._meta_cats[0].model.to(device)
        
    def process_entities(self, entities):
        results = []
        for v in entities['entities'].values():
            match = AnnotationMatch(
                start=v['start'],
                end=v['end'],
                term=v['detected_name'],
                snomed_id=v['cui'],
                similarity=v['context_similarity']
            )
            results.append(match)
        
        return results

    def annotate(self, text: str, return_ids_only = False):
        """
        Annotates a text to find all SNOMED concepts present in the text 

        Args:
            text: Text to annotate
            return_ids_only: Whether to return only the SNOMED ids present in the text or to return the `AnnotationMatch` objects too
        """
        ents = self.cat.get_entities(text, only_cui=return_ids_only)

        if return_ids_only:
            return list(ents['entities'].values())
        
        return self.process_entities(ents)
    
    def batch_annotate(self, texts: List[str], return_ids_only = False) -> List:
        """
        Annotates multiple texts to find all SNOMED concepts present in the texts
        
        Args:
            text: Text to annotate
            return_ids_only: Whether to return only the SNOMED ids present in the text or to return the `AnnotationMatch` objects too
        """
        results = self.cat.get_entities_multi_texts(texts, only_cui=return_ids_only)
        if return_ids_only:
            return list(map(lambda x: list(x['entities'].values()), results))

        return list(map(self.process_entities, results))
