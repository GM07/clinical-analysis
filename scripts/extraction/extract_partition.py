from argparse import ArgumentParser
import logging

from src.data.dataset import Dataset, DatasetPartition
from src.generation.ontology_beam_scorer import GenerationConfig
from src.pipelines.extraction_pipeline import PartitionedExtractionPipeline

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
parser.add_argument('--gen_config', type=str, help='Which config to use (normal, beam, constrained)')

configs = {
    'normal': GenerationConfig.greedy_search(),
    'beam': GenerationConfig.beam_search(),
    'constrained': GenerationConfig.ontology_beam_search()
}

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    config = args.gen_config

    assert config in ['normal', 'beam', 'constrained'], 'Generation config must be either normal, beam or constrained'
    gen_config = configs[config]

    pipeline = PartitionedExtractionPipeline(
        checkpoint_path=args.checkpoint,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
        medcat_path=args.medcat,
        medcat_device='cuda'
    )

    partition = DatasetPartition.from_save(args.partition)

    pipeline(partition, generation_config=gen_config)

if __name__ == '__main__':
    main()
