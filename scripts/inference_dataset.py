from argparse import ArgumentParser
import logging

from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline
from src.data.utils import rows_to_chat

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Tokenizer checkpoint')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset (csv)')
parser.add_argument('--output_path', type=str, required=True, help='Path where the output dataset will be saved')
parser.add_argument('--max_rows_to_process', type=int, default=100, help='Maximum number of rows to process')
parser.add_argument('--rows_to_chat', type=bool, default=True, help='Rows to chat')
parser.add_argument('--input_column', type=str, default='PROMPT', help='Column to use as input')
parser.add_argument('--output_column', type=str, default='OUTPUT', help='Column to use as output')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = ModelDatasetInferencePipeline(
        model_path=args.checkpoint,
        tokenizer_path=args.tokenizer
    )

    dataset = HuggingFaceDataset.from_csv(args.dataset)

    output_dataset = pipeline(
        dataset, 
        input_column=args.input_column, 
        output_column=args.output_column,
        max_rows_to_process=args.max_rows_to_process,
        rows_to_chat=rows_to_chat if args.rows_to_chat else None
    )

    output_dataset.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
