from argparse import ArgumentParser
import logging

from src.data.dataset import DatasetPartition
from src.pipelines.dataset_inference_pipeline import PartitionedInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--partition', type=str, required=True, help='Path to partition file')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--input_column', type=str, default='prompt', help='Column to use as input')
parser.add_argument('--chat_mode', type=bool, default=True, help='Whether to use chat mode')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = PartitionedInferencePipeline(
        model_path=args.checkpoint,
        input_column=args.input_column,
        chat_mode=args.chat_mode,
    )

    partition = DatasetPartition.from_save(args.partition)

    pipeline(partition, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

if __name__ == '__main__':
    main()
