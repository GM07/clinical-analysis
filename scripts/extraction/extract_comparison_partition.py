from argparse import ArgumentParser
import logging

from src.data.dataset import Dataset, DatasetPartition
from src.pipelines.extraction_pipeline import PartitionedComparisonExtractionPipeline, ExtractionPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--partition', type=str, required=True, help='Path to partition file')

parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--snomed', type=str, help='Path to snomed ontology file (.owx)')
parser.add_argument('--snomed_cache', type=str, help='Path to snomed cache file')
parser.add_argument('--medcat', type=str, help='Path to medcat annotator checkpoint')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = PartitionedComparisonExtractionPipeline(
        checkpoint_path=args.checkpoint,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
        medcat_path=args.medcat,
        medcat_device='cuda'
    )
    pipeline.load()

    partition = DatasetPartition.from_save(args.partition)

    pipeline(partition)

if __name__ == '__main__':
    main()
