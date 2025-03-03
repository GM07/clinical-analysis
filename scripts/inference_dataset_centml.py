from argparse import ArgumentParser
import asyncio
import logging

from datasets import Dataset as HuggingFaceDataset

from src.pipelines.dataset_inference_pipeline import OpenAIDatasetInferencePipeline
from src.data.utils import rows_to_chat

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--dataset', type=str, help='Path to dataset (csv)')
parser.add_argument('--api_key', type=str, help='API key')
parser.add_argument('--output_path', type=str, help='Path where the output dataset will be saved')
parser.add_argument('--max_rows_to_process', type=int, default=25000, help='Maximum number of rows to process')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

async def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    dataset = HuggingFaceDataset.from_csv(args.dataset)

    pipeline = OpenAIDatasetInferencePipeline(
        base_url='https://api.centml.com/openai/v1/', 
        api_key=args.api_key, 
        model_name='meta-llama/Llama-3.3-70B-Instruct'
    )

    await pipeline(
        dataset, 
        rows_to_chat=rows_to_chat, 
        max_rows_to_process=args.max_rows_to_process,
        batch_size=args.batch_size, 
        saving_path=args.output_path
    )

if __name__ == '__main__':
    asyncio.run(main())
