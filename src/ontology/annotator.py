
from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import logging

import spacy
from spacy import displacy
import nltk
import numpy as np
from medcat.cat import CAT
from sklearn.cluster import DBSCAN

import faiss
from transformers import pipeline

from src.ontology.snomed import Snomed
from src.ontology.embedding import OntologyEmbeddingRetriever

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


class OntologyEmbeddingAnnotator(Annotator):
    """
    Annotator of clinical notes based on embeddings of ontological concepts
    """

    def __init__(
        self, 
        embedding_model_path: str, 
        vector_db_path: str, 
        snomed_path: str, 
        snomed_cache_path: str,
    ):
        """
        Args:
            embedding_model_path: Path to model that will be used to generate embeddings
            vector_db_path: Path to vector database containing embeddings of ontological concepts
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache
        """
        super().__init__()

        self.embedding_model_path = embedding_model_path
        self.load_model()

        self.retriever = OntologyEmbeddingRetriever(
            vector_db_path=vector_db_path,
            snomed_path=snomed_path,
            snomed_cache_path=snomed_cache_path
        )

    def load_model(self):
        logger.info(f'Loading model located at {self.embedding_model_path}')
        self.extractor = pipeline(
            "feature-extraction",
            model=self.embedding_model_path,
            device='cuda',
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt',
            local_files_only=True
        )

    def preprocess(self, text: str):
        paragraphs = re.split('((\n\n)+|(  )+)', text)
        final_sentences = []
        start_ends = []
        for sentence in paragraphs:
            if sentence is None or len(sentence) == 0:
                continue

            sentences = nltk.sent_tokenize(sentence)
            for sentence in sentences:
                # sentence = re.sub(r'\n', ' ', sentence)
                if len(sentence) > 2:
                    final_sentences.append(sentence)
                    start = text.find(sentence)
                    start_ends.append((start, start + len(sentence)))

        return final_sentences, start_ends
    
    def get_word_sequences_with_spans(self, text, sequence_length=5, overlap_window=0):
        # Use spaCy's tokenization instead of split()
        nlp = spacy.blank('en')
        doc = nlp.make_doc(text)
        
        if len(doc) < sequence_length:
            return [], []
        
        step_size = sequence_length - overlap_window
        sequences = []
        spans = []
        
        for i in range(0, len(doc) - sequence_length + 1, step_size):
            # Get the tokens for this sequence
            sequence_tokens = doc[i:i + sequence_length]
            
            # Get start position of first token and end position of last token
            start_pos = sequence_tokens[0].idx
            end_pos = sequence_tokens[-1].idx + len(sequence_tokens[-1].text)
            
            # Join the tokens with space to create sequence
            sequence = ' '.join(token.text for token in sequence_tokens)
            
            sequences.append(sequence)
            spans.append((start_pos, end_pos))
        
        return sequences, spans


    def generate_embeddings(self, sentences, batch_size: int = 8):
        outputs = self.extractor(sentences, batch_size=batch_size)
        
        # Convert outputs to numpy array
        batch_vectors = np.vstack([output[0, 0] for output in outputs])
        batch_vectors = batch_vectors.astype(np.float32)
        
        return batch_vectors

    def cluster_embeddings(self, embeddings):
        """
        Clusters embeddings and returns the centroids of the clusters

        Args:
            embeddings: Numpy array of normalized embeddings that must be clustered
        """

        # Create distance matrix using cosine similarity
        # similarity_matrix = cosine_similarity(embeddings)
        # Convert similarity to distance (1 - similarity since cosine similarity ranges from -1 to 1)
        # distance_matrix = 1 - similarity_matrix
        # print(distance_matrix)

        # Initialize DBSCAN clusterer
        clusterer = DBSCAN(
            eps=0.3,
            min_samples=1,
            metric='cosine'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Get unique cluster labels
        unique_clusters = np.unique(labels)
        
        # Calculate centroids for each cluster
        centroids = []
        for cluster_id in unique_clusters:
            cluster_points = embeddings[labels == cluster_id]
            # For cosine similarity, we normalize the mean vector to maintain unit norm
            centroid = np.mean(cluster_points, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids.append(centroid)
        
        return np.array(centroids)

    def annotate(self, text: str, return_ids_only=False) -> List:
        sentences, start_ends = self.get_word_sequences_with_spans(text, sequence_length=10, overlap_window=0)

        embeddings = self.generate_embeddings(sentences)
        faiss.normalize_L2(embeddings)
        print('before : ', len(embeddings))
        # embeddings = self.cluster_embeddings(embeddings)
        print('after : ', len(embeddings))


        results = []
        for embedding, sentence, start_end in zip(embeddings, sentences, start_ends):
            concepts = self.retriever.find_similar(embedding, k=1)
            concept, similarity = concepts[0]

            if return_ids_only:
                results.append(concept)
            else:
                match = AnnotationMatch(
                    start=start_end[0],
                    end=start_end[1],
                    term=sentence,
                    snomed_id=concept,
                    similarity=similarity
                )
                results.append(match)
        return results

    def render(self, text, render_labels=False, snomed: Snomed = None):
        return self.annotate(text)


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
