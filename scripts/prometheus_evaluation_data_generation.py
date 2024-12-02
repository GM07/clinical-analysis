from argparse import ArgumentParser
import logging

import pandas as pd

from src.dataset import ExtractionDataset
from src.evaluation.prometheus import PrometheusPromptGenerator


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Generates the prometheus evaluation dataset from an extraction dataset')

parser.add_argument('--dataset', type=str, required=True, help='Path to extraction dataset')
parser.add_argument('--out', type=str, required=True, help='Path where the prometheus evaluation dataset will be generated')
parser.add_argument('--snomed', type=str, help='Path to snomed ontology')
parser.add_argument('--snomed_cache', type=str, help='Path to snomed cache')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    extraction_results = ExtractionDataset(args.dataset)

    evaluator = PrometheusPromptGenerator(
        args.snomed,
        args.snomed_cache
    )

    evaluator.generate_prompts(extraction_results, rubric='factuality', output_file_path=args.out)
    # prompts.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
