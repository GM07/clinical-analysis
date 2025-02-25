from argparse import ArgumentParser
import logging

from src.data.dataset import Dataset, DatasetPartition
from src.pipelines.extraction_pipeline import DatasetExtractionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset csv file')
parser.add_argument('--output_path', type=str, required=True, help='Path to output dataset csv file')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--snomed', type=str, help='Path to snomed ontology file (.owx)')
parser.add_argument('--snomed_cache', type=str, help='Path to snomed cache file')
parser.add_argument('--medcat', type=str, help='Path to medcat annotator checkpoint')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    print('Loading dataset from : ', args.dataset_path)
    dataset = Dataset.from_csv(args.dataset_path)

    pipeline = DatasetExtractionPipeline(
        checkpoint_path=args.checkpoint,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
        medcat_path=args.medcat,
        medcat_device='cuda'
    )
    print('Loading pipeline')
    pipeline.load()

    print('Extracting dataset')
    output_dataset = pipeline(dataset)
    
    print('Saving dataset to : ', args.output_path)
    output_dataset.to_csv(args.output_path)

if __name__ == '__main__':
    main()
