from argparse import ArgumentParser
import logging

from datasets import Dataset as HuggingFaceDataset
from datasets import load_from_disk

from src.models.openbiollm import OpenBioLLM
from src.models.prometheus import Prometheus
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
parser.add_argument('--system_prompt', type=str, default='none', required=False, help='System prompt to use (none, prometheus, bio, custom). If custom, use the format custom:<the prompt you want')

SYSTEM_PROMPTS = {
    'none': None,
    'prometheus': Prometheus.SYSTEM_PROMPT,
    'bio': OpenBioLLM.SYSTEM_PROMPT,
}

def get_system_prompt(prompt_type: str):

    if prompt_type.startswith('custom:'):
        return prompt_type.replace('custom:', '')

    return SYSTEM_PROMPTS[prompt_type]
    

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    system_prompt = get_system_prompt(args.system_prompt)

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
        system_prompt=system_prompt,
        **pipeline_args
    )

    output_dataset.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
