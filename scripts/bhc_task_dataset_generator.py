from argparse import ArgumentParser
import logging

from src.pipelines.bhc_task_dataset_pipeline import BhcTaskDatasetPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that loads the MIMIC-III clinical notes from\
                        the noteeevents.csv file, formats it and filters out invalid data.')

parser.add_argument('--notes_file', type=str, required=True, help='Path to noteevents.csv file of the MIMIC-III dataset')
parser.add_argument('--out', type=str, required=True, help='Output path where the processed dataset will be saved')
parser.add_argument('--tokenizer', type=str, required=True, help='Path to huggingface tokenizer used to filter out')
parser.add_argument('--max_notes', type=str, required=True, help='Max number of notes tha the processed dataset can have')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = BhcTaskDatasetPipeline(
        mimic_path=args.notes_file,
        output_file_path=args.out,
        tokenizer_path=args.tokenizer,
        max_total_nb_notes=int(args.max_notes)
    )

    pipeline()

if __name__ == '__main__':
    main()
