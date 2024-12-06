from argparse import ArgumentParser
import logging

from src.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that merges partitions into a single csv file')

parser.add_argument('--partition_folder', type=str, required=True, help='Path to partition folder')
parser.add_argument('--out', type=str, required=True, help='Path to output file')
parser.add_argument('--results', type=str, required=True, help='Name of columns in csv file for the results per sample. Must be a string of comma-separated names')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    column_names = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), args.results.split(','))))
    # print(column_names)
    Dataset.partitions_to_file(args.partition_folder, output_file_path=args.out, column_names=column_names)

if __name__ == '__main__':
    main()
