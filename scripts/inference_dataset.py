from argparse import ArgumentParser
import logging

from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.pipelines.dataset_inference_pipeline import HuggingFaceDatasetInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--dataset', type=str, help='Path to dataset (csv)')
parser.add_argument('--input_column', type=str, default='PROMPT', help='Column to use as input')
parser.add_argument('--output_column', type=str, default='OUTPUT', help='Column to use as output')
parser.add_argument('--output_path', type=str, help='Path where the output dataset will be saved')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = HuggingFaceDatasetInferencePipeline(
        model_path=args.checkpoint,
        input_column=args.input_column,
        output_column=args.output_column,
    )

    dataset = HuggingFaceDataset.from_csv(args.dataset)

    pipeline(dataset)

    dataset.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
