

from argparse import ArgumentParser
import logging

from src.data.dataset import ComparisonExtractionDataset
from src.evaluation.factuality_evaluator.extraction_factuality_evaluator import ExtractionFactualityEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Evaluates the extractions using a MedHal model')

parser.add_argument('--model', required=True, type=str, help='Path to MedHal model')
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset containing the extractions')
parser.add_argument('--method', required=True, type=str, default='constrained', help='Name of the method to evaluate (normal, beam, constrained)')
parser.add_argument('--snomed', required=True, type=str, help='Path to snomed ontology file')
parser.add_argument('--snomed_cache', required=True, type=str, help='Path to snomed cache file')
parser.add_argument('--out', required=False, type=str, help='Path to csv file which will contain the results')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    dataset = ComparisonExtractionDataset(args.dataset)

    evaluator = ExtractionFactualityEvaluator(
        model_path=args.model,
        tokenizer_path=args.model,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
    )

    results = evaluator(extraction_dataset=dataset, method=args.method, out=args.out)
    print('Results : ', results)

if __name__ == '__main__':
    main()
