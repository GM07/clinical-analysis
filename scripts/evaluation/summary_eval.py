

from argparse import ArgumentParser
import logging

from datasets import Dataset
from src.evaluation.factuality_evaluator.summary_factuality_evaluator import SummaryFactualityEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Evaluates the extractions using a MedHal model')

parser.add_argument('--model', required=True, type=str, help='Path to MedHal model')
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset containing the extractions')
parser.add_argument('--out', required=False, type=str, help='Path to csv file which will contain the results')
parser.add_argument('--text_column', required=False, default='TEXT', type=str, help='Name of the column containing the text of the summaries')
parser.add_argument('--summary_column', required=False, default='SUMMARY', type=str, help='Name of the column containing the summary of the text')
parser.add_argument('--id_column', required=False, default='ROW_ID', type=str, help='Name of the column containing the id of the summaries')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    evaluator = SummaryFactualityEvaluator(
        model_path=args.model,
        tokenizer_path=args.model,
    )

    dataset = Dataset.from_csv(args.dataset)
    results = evaluator(dataset=dataset, out=args.out)
    print('Results : ', results)

if __name__ == '__main__':
    main()
