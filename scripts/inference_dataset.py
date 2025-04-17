from argparse import ArgumentParser
import logging

from datasets import Dataset as HuggingFaceDataset
from datasets import load_from_disk

from src.pipelines.dataset_inference_pipeline import HFModelDatasetInferencePipeline, ModelDatasetInferencePipeline
from src.data.utils import rows_to_chat

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Tokenizer checkpoint')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset (csv or huggingface on disk)')
parser.add_argument('--output_path', type=str, required=True, help='Path where the output dataset will be saved')
parser.add_argument('--max_rows_to_process', type=int, default=100, help='Maximum number of rows to process')
parser.add_argument('--rows_to_chat', type=bool, default=True, help='Rows to chat')
parser.add_argument('--input_column', type=str, default='PROMPT', help='Column to use as input')
parser.add_argument('--output_column', type=str, default='OUTPUT', help='Column to use as output')
parser.add_argument('--apply_chat_template', type=bool, default=True, help='Whether to apply the chat template or not')
parser.add_argument('--hf', type=bool, default=False, required=False, help='Whether to use HuggingFace for inference (if False, will use vLLM)')
parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size to use (if using Huggingface for inference)')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline_args = {}
    if args.hf:
        pipeline = HFModelDatasetInferencePipeline(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )
        pipeline_args['batch_size'] = args.batch_size
    else:
        pipeline = ModelDatasetInferencePipeline(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )

    dataset_path: str = args.dataset

    if dataset_path.endswith('.csv'):
        dataset = HuggingFaceDataset.from_csv(args.dataset)
    else:
        dataset = load_from_disk(dataset_path)
        print(dataset)

    output_dataset = pipeline(
        dataset, 
        input_column=args.input_column, 
        output_column=args.output_column,
        max_rows_to_process=args.max_rows_to_process,
        rows_to_chat=rows_to_chat if args.rows_to_chat else None,
        apply_chat_template=args.apply_chat_template,
        # system_prompt="You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."
        **pipeline_args
    )

    output_dataset.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
