from argparse import ArgumentParser
import logging

from src.data.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that divides a dataset into multiple partitions')

parser.add_argument('--dataset', type=str, required=True, help='Path to .csv file of the dataset')
parser.add_argument('--out', type=str, required=True, help='Output path where the partitioned dataset will be saved')
parser.add_argument('--size', type=str, required=True, help='Size of a partition')
parser.add_argument('--max_rows', type=int, required=False, help='Max number of rows to be partitioned')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    dataset = Dataset(args.dataset)
    dataset.partition(output_folder_path=args.out, size_of_partition=int(args.size), max_rows=int(args.max_rows) if args.max_rows else None)

if __name__ == '__main__':
    main()
