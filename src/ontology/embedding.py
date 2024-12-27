import logging
import os
from tqdm import tqdm
from typing import List

from transformers import pipeline
import faiss
import sqlite3
import numpy as np
import pandas as pd

import umap
import umap.plot as uplot

from src.ontology.snomed import Snomed
from src.utils import batch_elements
from src.ontology.ontology_filter import OntologyFilter, BranchFilter

logger = logging.getLogger(__name__)

class OntologyEmbeddingBase:
    def __init__(self, vector_db_path: str, snomed_path: str, snomed_cache_path: str):
        """
        Args:
            vector_db_path: Path to vector database folder containing `embeddings.db` and `embeddings.index`.
            These files can be generated from the snomed ontology and a model using `OntologyEmbeddingRetriever.generate_ontology_embedding()`
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache
        """
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path
        self.vector_db_path = vector_db_path
        self.index_file = vector_db_path + OntologyEmbeddingRetriever.EMBEDDINGS_INDEX_PATH
        self.db_file = vector_db_path + OntologyEmbeddingRetriever.EMBEDDINGS_DATABASE_PATH

        assert os.path.exists(self.index_file), f'No such index file {self.index_file}'
        assert os.path.exists(self.db_file), f'No such database file {self.db_file}'

        self.load()

    def load(self):
        """
        Loads the FAISS index, the sqlite3 database and the snomed ontology
        """
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path)
        self.index = faiss.read_index(self.index_file)
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

    def get_all_embeddings(self):
        """Returns all embeddings in the retriever"""
        return self.index.reconstruct_n(0, self.index.ntotal)

    def get_embeddings_batch(self, start_idx, batch_size):
        """Extract a batch of embeddings from FAISS index"""
        end_idx = min(start_idx + batch_size, self.index.ntotal)
        batch_size = end_idx - start_idx
        
        batch_embeddings = np.empty((batch_size, self.index.d), dtype=np.float32)
        for i in range(batch_size):
            batch_embeddings[i] = self.index.reconstruct(start_idx + i)
            
        return batch_embeddings

    def get_all_concept_ids(self):
        self.cursor.execute('SELECT id FROM embeddings_map ORDER BY index_position')
        return [row[0] for row in self.cursor.fetchall()]

    def get_embedding(self, id):
        """
        Returns the embedding of a concept id
        """
        # Get the FAISS index position for this ID
        self.cursor.execute('SELECT index_position FROM embeddings_map WHERE id = ?', (id,))
        result = self.cursor.fetchone()
        
        if result is None:
            raise KeyError(f"ID {id} not found in database")
            
        return self.index.reconstruct(int(result[0]))
    
    def get_multiple_embeddings(self, ids: List[str], batch_size: int = 30000):
        """
        Returns embeddings of multiple concept ids, handling large numbers of IDs
        by batching SQL queries.

        Args:
            ids: List of id's embeddings needed
            batch_size: Number of elements per SQL queries (maximum is 32766 for SQLite 3.32.0)
        """
        
        # Initialize collections for results
        all_results = []
        
        id_batches = batch_elements(ids, batch_size=batch_size)
        
        # Process each batch
        for id_batch in id_batches:
            placeholders = ','.join('?' * len(id_batch))
            query = f'SELECT id, index_position FROM embeddings_map WHERE id IN ({placeholders})'
            self.cursor.execute(query, id_batch)
            batch_results = self.cursor.fetchall()
            all_results.extend(batch_results)
        
        if not all_results:
            raise KeyError(f"None of the provided IDs were found")
        
        # Create a mapping of ID to index position
        id_to_position = {id: pos for id, pos in all_results}
        
        # Get embeddings for all found IDs
        indices = np.array([id_to_position[id] for id in ids if id in id_to_position], dtype=np.int64)
        embeddings = np.vstack([self.index.reconstruct(int(idx)) for idx in indices])
        
        # Return dictionary mapping IDs to their embeddings
        return {id: embeddings[i] for i, id in enumerate(ids) if id in id_to_position}

    def close(self):
        self.conn.close()


    @staticmethod
    def generate_ontology_embeddings(
        model_path: str,
        snomed_path: str, 
        snomed_cache_path: str, 
        vector_db_out_path: str, 
        batch_size: int,
        hidden_size: int = 1024
    ):
        if vector_db_out_path[-1] != '/':
            vector_db_out_path += '/'

        index_file = vector_db_out_path + OntologyEmbeddingRetriever.EMBEDDINGS_INDEX_PATH
        logger.info(f'Index file : {index_file}')
        
        db_file = vector_db_out_path + OntologyEmbeddingRetriever.EMBEDDINGS_DATABASE_PATH
        logger.info(f'DB file : {db_file}')

        # Overwritting vector database files if already present
        if not os.path.exists(vector_db_out_path):
            os.mkdir(vector_db_out_path)

        if os.path.exists(db_file):
            os.remove(db_file)

        if os.path.exists(index_file):
            os.remove(index_file)

        logger.info('Loading embedding model...')
        extractor = pipeline(
            "feature-extraction",
            model=model_path,
            device='cuda',
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            local_files_only=True
        )

        logger.info('Generating description of ontological concepts...')
        snomed = Snomed(snomed_path, snomed_cache_path)

        all_ids = list(snomed.id_to_classes.keys()) # TODO : Filter branches of ontology
        all_texts = list(map(lambda x: snomed.get_contextual_description_of_id(x), all_ids))

        batched_ids = batch_elements(all_ids, batch_size)
        batched_texts = batch_elements(all_texts, batch_size)

        logger.info('Creating index and database...')

        index = faiss.IndexFlatIP(hidden_size)
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings_map
                    (id TEXT PRIMARY KEY, index_position INTEGER)''')

        logger.info('Filling database and index...')

        current_index = 0
        for ids, texts in tqdm(zip(batched_ids, batched_texts), total=len(batched_ids)):
            outputs = extractor(texts, batch_size=batch_size)
            
            # Convert outputs to numpy array
            batch_vectors = np.vstack([output[0, 0] for output in outputs])
            batch_vectors = batch_vectors.astype(np.float32)
            
            # Add to FAISS index
            faiss.normalize_L2(batch_vectors)
            index.add(batch_vectors)
            
            # Store ID mapping in SQLite
            mapping_data = [(id, current_index + i) for i, id in enumerate(ids)]
            c.executemany('INSERT INTO embeddings_map VALUES (?, ?)', mapping_data)
            current_index += len(ids)
            
            conn.commit()

        # Save FAISS index
        faiss.write_index(index, index_file)
        conn.close()


class OntologyEmbeddingVisualizer(OntologyEmbeddingBase):
    """
    Helper to visualize embeddings generated from ontology concepts
    """

    def __init__(self, vector_db_path, snomed_path, snomed_cache_path):
        """
        Args:
            vector_db_path: Path to vector database folder containing `embeddings.db` and `embeddings.index`.
            These files can be generated from the snomed ontology and a model using `OntologyEmbeddingRetriever.generate_ontology_embedding()`
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache
        """
        super().__init__(vector_db_path, snomed_path, snomed_cache_path)

        # Caching
        self.reducer = None


    def get_all_labels(self):
        ids = self.get_all_concept_ids()
        return self.snomed.convert_ids_to_labels(ids)

    def get_all_primary_concepts(self, ids: List[str]):
        """
        Returns the primary concepts of a set of ids

        Args:
            ontology_filter: Filter that will remove some ids in the ontology
        
        """
        concept = []
        primary_concepts = set(Snomed.PRIMARY_CONCEPTS)
        for id in tqdm(ids, total=len(ids)):
            if id in primary_concepts:
                concept.append(id)
                continue

            ancestors: set = self.snomed.get_ancestors_of_id(id, return_set=True)
            intersect = ancestors.intersection(primary_concepts)
            if len(intersect) == 0:
                concept.append(Snomed.BASE_CLASS_ID)
            else:
                concept.append(list(intersect)[0])

        return concept
    
    def reduce_embeddings(self, n_components: int = 2, use_cache: bool = True, ids: List[str] = None):
        """
        Performs dimensionality reduction on all embeddings using UMAP

        Args:
            n_components: Number of dimensions after dimensionality reduction
            use_cache: Whether to overwrite the current embeddings generated (takes a lot of time to regenerate the embeddings)
            ids: List of ids to consider (if not provided, it will consider all ids)
        """
        if use_cache and self.reducer is not None:
            return self.reducer.embedding_
        
        ids = self.get_all_concept_ids() if ids is None else ids
        
        logger.info(f'Filter resulted in {len(ids)} concepts')
        embeddings = self.get_multiple_embeddings(ids) # {id: embedding}

        logger.info("Computing UMAP...")
        self.reducer = umap.UMAP(
            n_components=n_components,
            # low_memory=True,
            # verbose=True,
            metric='cosine'
        )

        embeddings_2d = self.reducer.fit_transform(np.array(list(embeddings.values())))

        return embeddings_2d

    def visualize_embeddings(self, n_components: int = 2, ontology_filter: OntologyFilter = None, use_cache: bool = True):
        """
        Visualizes embeddings in a graph by reducing the embedding dimension to 2d with UMAP

        Args:
            n_components: Number of dimensions after dimensionality reduction
            ontology_filter: Filter that will remove some ids in the ontology
            use_cache: Whether to overwrite the current embeddings generated (takes a lot of time to regenerate the embeddings)
        """
        ids = self.get_all_concept_ids if ontology_filter is None else ontology_filter(self.get_all_concept_ids())
        self.reduce_embeddings(n_components=n_components, ids=ids, use_cache=use_cache)

        labels = self.snomed.convert_ids_to_labels(ids)        
        clusters = self.get_all_primary_concepts(ids)
        
        # Color is based on the primary concept of Snomed
        cluster_labels = list(map(lambda x: self.snomed.get_label_from_id(x), self.snomed.PRIMARY_CONCEPTS))
        label_mapping = dict(zip(self.snomed.PRIMARY_CONCEPTS, cluster_labels))
        
        self.hover_data = pd.DataFrame({'index': ids, 'label': labels, 'primary_concept_id': clusters})
        self.hover_data['cluster'] = self.hover_data.primary_concept_id.map(label_mapping)

        uplot.output_notebook()
        p = uplot.interactive(self.reducer, labels=self.hover_data['cluster'], hover_data=self.hover_data, width=1900, height=1900, point_size=3)
        return p
    
class OntologyEmbeddingRetriever(OntologyEmbeddingBase):
    """
    Class that uses embeddings of ontological concepts to retrieve most similar concepts given a query.
    """

    EMBEDDINGS_DATABASE_PATH: str = 'embeddings.db'
    EMBEDDINGS_INDEX_PATH: str = 'embeddings.index'

    def __init__(self, vector_db_path: str, snomed_path: str, snomed_cache_path: str):
        """
        Args:
            vector_db_path: Path to vector database folder containing `embeddings.db` and `embeddings.index`.
            These files can be generated from the snomed ontology and a model using `OntologyEmbeddingRetriever.generate_ontology_embedding()`
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache
        """
        super().__init__(vector_db_path, snomed_path, snomed_cache_path)

        # Caching
        self.embeddings_2d = None
        self.cluster_labels = None
        self.ids = None
        self.reducer = None


    def find_similar(self, query_embedding, k=5):
        """
        Find k most similar embeddings using cosine similarity
        
        Args:
            query_embedding: Embedding numpy array of shape (1024,) representing a query
            k: Number of similar embeddings to return
        
        Returns:
            List of tuples (id, similarity_score)
            Note: similarity_score will be between -1 and 1, where 1 is most simila r
        """
        # Reshape and normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search the index
        similarities, indices = self.index.search(query, k)
        
        # Get IDs for these indices
        similar_items = []
        for idx, sim in zip(indices[0], similarities[0]):
            self.cursor.execute('SELECT id FROM embeddings_map WHERE index_position = ?', (int(idx),))
            result = self.cursor.fetchone()
            if result:
                # Note: similarity score is already between -1 and 1
                similar_items.append((result[0], float(sim)))
    
        return similar_items
