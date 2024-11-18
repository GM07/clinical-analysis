from argparse import ArgumentParser
import logging
import subprocess

from src.dataset import Dataset, DatasetPartition, DatasetPartitionAnalyzer

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='This program will call a script for a certain number of jobs on all unprocessed partitions in an partition folder')

parser.add_argument('--program', type=str, required=True, help='Path to program that will be queued up in the scheduler. \
                    Note that this script must take as a first argument the path to the partition and optionally, as a second argument, the path to the model checkpoint')

parser.add_argument('--nb_jobs', type=int, required=True, help='Number of jobs that should be scheduled')
parser.add_argument('--partition_folder', type=str, required=True, help='Path to folder containing all partition files')
parser.add_argument('--model', type=str, required=False, help='Path to model checkpoint. If not provided, will not be sent to the program')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    analyzer = DatasetPartitionAnalyzer(args.partition_folder)
    partition_files = analyzer.get_next_partitions(args.nb_jobs)

    model_arg = args.model if args.model else ''
    for partition_file in partition_files:
        command = ['sbatch', f'{args.program}', f'{partition_file}', model_arg]
        print(command)
        subprocess.run(command)


if __name__ == '__main__':
    main()
