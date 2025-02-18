from argparse import ArgumentParser
import logging

from src.domain_adaptation.evaluator_dataset import EvaluatorDatasetSummarizer

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--dataset_path', type=str, required=True, help='Path to clinical notes of each domain')
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--output_path', type=str, required=True, help='Path to output file')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)
    summarizer = EvaluatorDatasetSummarizer(
        dataset_path=args.dataset_path,
        model_checkpoint=args.model_checkpoint
    )

    dataset = summarizer.summarize()

    dataset.to_csv(args.output_path)


if __name__ == '__main__':
    main()
