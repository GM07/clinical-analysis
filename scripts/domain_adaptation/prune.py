from argparse import ArgumentParser
import logging
from typing import List

from src.data.dataset import ExtractionDataset
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.domain_adaptation.pruner import Pruner
from src.ontology.snomed import Snomed

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that prunes the extractions of a dataset. \
                        The dataset will be saved in the same file as the input file, but with a new column containing the pruned extractions.')

parser.add_argument('--dataset', type=str, required=True, help='Path to extraction dataset file (csv file)')
parser.add_argument('--input_column', type=str, required=True, help='Column containing the extractions (pruned version will be named based on this column)')
parser.add_argument('--snomed', type=str, required=True, help='Path to SNOMED file (owl file)')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to SNOMED cache file')
parser.add_argument('--dcf_files', type=str, nargs='+', required=True, help='Path to DCF files (one per domain)')
parser.add_argument('--alpha', type=int, default=1, help='Alpha value')
parser.add_argument('--output_dataset', type=str, required=True, help='Path to output dataset file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    dataset = ExtractionDataset(column=args.input_column, dataset_path=args.dataset)
    snomed = Snomed(args.snomed, args.snomed_cache)
    for dcf_path in args.dcf_files:
        dcf = DomainClassFrequency.load(dcf_path)
        pruner = Pruner(dcf, snomed)
        pruner.prune_dataset(dataset, args.input_column, args.alpha)
    
    dataset.save(args.output_dataset)

if __name__ == '__main__':
    main()
