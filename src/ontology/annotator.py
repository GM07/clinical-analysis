
from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from medcat.cat import CAT
import spacy
from spacy import displacy

from src.ontology.snomed import Snomed

@dataclass
class AnnotationMatch:

    start: int
    end: int
    term: str
    snomed_id: str
    similarity: float

class Annotator(ABC):
    """Base class for annotating texts"""
    
    @abstractmethod
    def annotate(self, text: str, return_ids_only = False) -> List:
        """
        Annotates a text to find all SNOMED concepts present in the text 

        Args:
            text: Text to annotate
            return_ids_only: Whether to return only the SNOMED ids present in the text or to return the `AnnotationMatch` objects too
        """
        pass

    def batch_annotate(self, texts: List[str], return_ids_only = False) -> List:
        """
        Annotates multiple texts to find all SNOMED concepts present in the texts
        
        Args:
            text: Text to annotate
            return_ids_only: Whether to return only the SNOMED ids present in the text or to return the `AnnotationMatch` objects too
        """
        results = []
        for text in texts:
            results.append(self.annotate(text, return_ids_only=return_ids_only))
        return results

    @abstractmethod
    def render(self, text, render_labels=False, snomed: Snomed = None):
        """
        Annotates and renders the annotated text using displacy

        Args:
            text: Text to annotate and render
            render_labels: Whether to render the labels the concepts or their ids. If True, `snomed` must be provided.
            snomed: Snomed instance to find the labels of the concepts. Needed when `render_snomed_ids` = True
        """
        pass

class EmbeddingAnnotator(Annotator):
    """
    Annotator of clinical notes based on an embedding vector
    """

    def __init__(self):
        super().__init__()

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

    def render(self, text, render_labels=False, snomed: Snomed = None):
        """
        Annotates and renders the annotated text using displacy

        Args:
            text: Text to annotate and render
            render_labels: Whether to render the labels the concepts or their ids. If True, `snomed` must be provided.
            snomed: Snomed instance to find the labels of the concepts. Needed when `render_snomed_ids` = True
        """
        if render_labels:
            assert snomed is not None, 'Rendering option set to concept labels, but a snomed instance was not provided in the call render()'

        nlp = spacy.blank('en')
        doc = nlp.make_doc(text)
        results: List[AnnotationMatch] = self.annotate(text)
        ents = []
        for result in results:
            if render_labels:
                label = 'N/A'
                if snomed.is_id_valid(result.snomed_id):
                    label = snomed.get_label_from_id(result.snomed_id)
                ent = doc.char_span(int(result.start), int(result.end), label=label)
            else:
                ent = doc.char_span(int(result.start), int(result.end), label=result.snomed_id)
            if ent is not None:
                ents.append(ent)
        doc.ents = ents
        return displacy.render(doc, style='ent')
