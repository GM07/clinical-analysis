from argparse import ArgumentParser
import logging

from src.data.dataset import DatasetPartitionAnalyzer

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--partition_folder', type=str, required=True, help='Path to partition folder')
parser.add_argument('--reduce_factor', type=int, required=False, default=5, help='Reduce factor when showing the partitions')
parser.add_argument('--fix', type=bool, required=False, default=False, help='Whether to fix broken partitions or not')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)


    analyzer = DatasetPartitionAnalyzer(args.partition_folder)
    analyzer.show_partitions(reduce_factor=5)

    if args.fix:
        analyzer.fix_broken_partitions()

if __name__ == '__main__':
    main()
