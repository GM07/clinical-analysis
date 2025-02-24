from argparse import ArgumentParser
import logging
from typing import List

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.domain_adaptation.pruner import Pruner
from src.ontology.snomed import Snomed
from src.domain_adaptation.verbalizer import Verbalizer
from src.data.dataset import PrunedConceptDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that prunes the extractions of a dataset. \
                        The dataset will be saved in the same file as the input file, but with a new column containing the pruned extractions.')

parser.add_argument('--dataset', type=str, required=True, help='Path to extraction dataset file (csv file)')
parser.add_argument('--model_path', type=str, required=True, help='Path to huggingface model file')
parser.add_argument('--snomed', type=str, required=True, help='Path to SNOMED file (owl file)')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to SNOMED cache file')
parser.add_argument('--output_dataset', type=str, required=True, help='Path to output dataset file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    snomed = Snomed(args.snomed, args.snomed_cache)

    columns = ['constrained_ecg', 'constrained_nursing_other', 'constrained_radiology']
    verbalizer = Verbalizer(model_path=args.model_path, input_columns=columns, snomed=snomed)

    pruned_dataset = PrunedConceptDataset(columns=columns, dataset_path=args.dataset)
    # logger.info(f'Verbalizing dataset with {len(pruned_dataset)} rows and columns : {pruned_dataset.column_names}')

    # for dcf_path in args.dcf_files:
        # dcf = DomainClassFrequency.load(dcf_path)
        # pruner = Pruner(dcf, snomed)
        # pruner.prune_dataset(dataset, args.input_column, args.alpha)
    
    # dataset.save(args.output_dataset)

if __name__ == '__main__':
    main()
