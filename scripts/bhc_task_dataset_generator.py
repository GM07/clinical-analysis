from argparse import ArgumentParser
import logging

from src.pipelines.bhc_task_dataset import BhcTaskDatasetPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

# python scripts/bhc_task_dataset_generator.py --notes_file /home/gmehenni/projects/def-azouaq/gmehenni/data/mimic/raw/noteevents.csv --out /home/gmehenni/scratch/mimic_processed.csv --tokenizer /home/gmehenni/projects/def-azouaq/gmehenni/models/Meta-Llama-3-8B-Instruct
parser = ArgumentParser(description='Program that loads the MIMIC-III clinical notes from\
                        the noteeevents.csv file, formats it and filters out invalid data. \
                        This will generate a dataset where each note')

parser.add_argument('--notes_file', type=str, required=True, help='Path to noteevents.csv file of the MIMIC-III dataset')
parser.add_argument('--out', type=str, required=True, help='Output path where the processed dataset will be saved')
parser.add_argument('--tokenizer', type=str, required=True, help='Path to huggingface tokenizer used to filter out')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = BhcTaskDatasetPipeline(
        mimic_path=args.notes_file,
        output_file_path=args.out,
        tokenizer_path=args.tokenizer
    )

    pipeline()

if __name__ == '__main__':
    main()
