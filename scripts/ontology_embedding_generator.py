from argparse import ArgumentParser
import logging

from src.ontology.embedding import OntologyEmbeddingBase

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that generates a vector database containing the \
                        embeddings of all ontological concepts in the snomed ontology using \
                        an embedding model')

parser.add_argument('--out', type=str, required=True, help='Output path where the vector database will be stored')
parser.add_argument('--checkpoint', type=str, required=True, help='Embedding Model checkpoint')
parser.add_argument('--snomed', type=str, required=True, help='Path to snomed ontology file (.owx)')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to snomed cache file')
parser.add_argument('--batch_size', type=int, required=False, default=640, help='Path to snomed cache file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    OntologyEmbeddingBase.generate_ontology_embeddings(
        model_path=args.checkpoint,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
        vector_db_out_path=args.out,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
