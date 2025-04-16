from argparse import ArgumentParser
import logging

from src.data.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that merges partitions into a single csv file')

parser.add_argument('--partition_folder', type=str, required=True, help='Path to partition folder')
parser.add_argument('--out', type=str, required=True, help='Path to output file')
parser.add_argument('--results', type=str, required=True, help='Name of columns in csv file for the results per sample. Must be a string of comma-separated names')
parser.add_argument('--original', type=str, required=False, default=None, help='Path to original dataset')
parser.add_argument('--preprocess', type=str, required=False, default='none', help='Path to original dataset (dict_to_list or none)')

def dict_to_list(d: dict):
    return [(k, v) for k, v in d.items()]

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    column_names = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), args.results.split(','))))
    print(column_names)
    Dataset.partitions_to_file(
        args.partition_folder, 
        output_file_path=args.out,
        column_names=column_names,
        original_dataset_path=args.original,
        preprocess=dict_to_list if args.preprocess == 'dict_to_list' else 'none',
    )

if __name__ == '__main__':
    main()
