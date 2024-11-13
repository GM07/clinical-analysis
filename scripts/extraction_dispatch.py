from argparse import ArgumentParser
import logging
import subprocess

from src.dataset import Dataset, DatasetPartition, DatasetPartitionAnalyzer

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='This program will call the extraction script for a certain number of jobs on all unprocessed partitions')

parser.add_argument('--extraction', type=str, required=True, help='Path to extraction script that will queue up a job in the scheduler. \
                    Note that this script must take as an argument the path to the partition')

parser.add_argument('--nb_jobs', type=int, required=True, help='Number of jobs that should be scheduled')
parser.add_argument('--partition_folder', type=str, required=True, help='Path to folder containing all partition files')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    analyzer = DatasetPartitionAnalyzer(args.partition_folder)
    partition_files = analyzer.get_next_partitions(args.nb_jobs)

    for partition_file in partition_files:
        command = ['sbatch', f'{args.extraction}', f'{partition_file}']
        print(command)
        subprocess.run(command)


if __name__ == '__main__':
    main()
