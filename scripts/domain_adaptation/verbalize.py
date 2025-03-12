from argparse import ArgumentParser
import logging
from typing import List

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.domain_adaptation.pruner import Pruner
from src.ontology.snomed import Snomed
from src.domain_adaptation.verbalizer import Verbalizer
from src.data.dataset import PrunedConceptDataset
from src.generation.templates import DEFAULT_SYSTEM_ENTRY, LLAMA_BIO_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that prunes the extractions of a dataset. \
                        The dataset will be saved in the same file as the input file, but with a new column containing the pruned extractions.')

parser.add_argument('--dataset', type=str, nargs='+', required=True, help='Path to extraction dataset file (csv file)')
parser.add_argument('--model_path', type=str, required=True, help='Path to huggingface model file')
parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to huggingface tokenizer file')
parser.add_argument('--snomed', type=str, required=True, help='Path to SNOMED file (owl file)')
parser.add_argument('--snomed_cache', type=str, required=True, help='Path to SNOMED cache file')
parser.add_argument('--output_dataset', type=str, nargs='+', required=True, help='Path to output dataset file')
parser.add_argument('--system_prompt', type=str, default='default', help='System prompt to use for the inference (default or bio)')
def main():

    args = parser.parse_args()
    print('Called with arguments : ', args)

    if args.system_prompt not in ['default', 'bio']:
        raise ValueError('System prompt must be either default or bio')

    system_prompt = DEFAULT_SYSTEM_ENTRY if args.system_prompt == 'default' else LLAMA_BIO_SYSTEM_PROMPT
    print('Using system prompt : ', system_prompt)

    assert len(args.dataset) == len(args.output_dataset), 'The number of dataset and output dataset must be the same'

    snomed = Snomed(args.snomed, args.snomed_cache)

    columns = ['constrained_ecg', 'constrained_nursing', 'constrained_radiology', 'constrained_physician_']
    verbalizer = Verbalizer(model_path=args.model_path, tokenizer_path=args.tokenizer_path, input_columns=columns, snomed=snomed)

    for dataset_path, output_path in zip(args.dataset, args.output_dataset):
        pruned_dataset = PrunedConceptDataset(columns=columns, dataset_path=dataset_path)
        verbalized_dataset = verbalizer.verbalize_dataset(pruned_dataset, system_prompt)
        verbalized_dataset.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
