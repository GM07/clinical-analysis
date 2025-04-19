
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


from src.ontology.snomed import Snomed

logger = logging.getLogger(__name__)

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

    def render(self, text, render_labels=False, snomed: Snomed = None):
        """
        Annotates and renders the annotated text using displacy

        Args:
            text: Text to annotate and render
            render_labels: Whether to render the labels the concepts or their ids. If True, `snomed` must be provided.
            snomed: Snomed instance to find the labels of the concepts. Needed when `render_snomed_ids` = True
        """
        import spacy
        from spacy import displacy

        if render_labels:
            assert snomed is not None, 'Rendering option set to concept labels, but a snomed instance was not provided in the call render()'

        nlp = spacy.blank('en')
        doc = nlp.make_doc(text)
        results: List[AnnotationMatch] = self.annotate(text)
        ents = []
        for result in results:
            if render_labels:
                label = snomed.get_label_from_id(result.snomed_id) if snomed.is_id_valid(result.snomed_id) else 'N/A'
                final_label = f'{label} ({result.similarity:.2f})'
            else:
                final_label = f'{result.snomed_id} ({result.similarity:.2f})'
            
            ent = doc.char_span(int(result.start), int(result.end), label=final_label)

            if ent is not None:
                ents.append(ent)
        doc.ents = ents
        return displacy.render(doc, style='ent')
