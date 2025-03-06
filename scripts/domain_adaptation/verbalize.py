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
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to huggingface tokenizer file')
parser.add_argument('--snomed', type=str, required=True, help='Path to SNOMED file (owl file)')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to SNOMED cache file')
parser.add_argument('--output_dataset', type=str, required=True, help='Path to output dataset file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    snomed = Snomed(args.snomed, args.snomed_cache)

    columns = ['constrained_ecg', 'constrained_nursing', 'constrained_radiology', 'constrained_physician_']
    verbalizer = Verbalizer(model_path=args.model_path, tokenizer_path=args.tokenizer_path, input_columns=columns, snomed=snomed)

    pruned_dataset = PrunedConceptDataset(columns=columns, dataset_path=args.dataset)
    verbalized_dataset = verbalizer.verbalize_dataset(pruned_dataset)
    verbalized_dataset.to_csv(args.output_dataset, index=False)

if __name__ == '__main__':
    main()
