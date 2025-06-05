from argparse import ArgumentParser
import logging

import pandas as pd

from src.data.dataset import Dataset
from src.data.med_hal.augmented_clinical_notes import AugmentedClinicalNotes
from src.data.med_hal.medmcqa import MedMCQA
from src.data.med_hal.mednli import MedNLI
from src.data.med_hal.medqa import MedQA
from src.data.med_hal.sumpubmed import SumPubMed

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description='Program that generates prompts from a dataset that is used to generate the Medical Hallucination Dataset (MHD) \
                        that will be sent to a bigger model that will generate samples for MHD')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset that is used to generate MHD')
parser.add_argument('--dataset', type=str, required=True, help='Type of dataset that is used to generate MHD (options: MedQA, MedMCQA, AugmentedClinicalNotes, MedNLI, SumPubmed)')
parser.add_argument('--out', type=str, required=True, help='Output path where the prompts will be saved')
parser.add_argument('--partition', type=bool, required=False, default=False, help='If True, the prompts will be saved in a partition file')
parser.add_argument('--size', type=int, required=False, default=1000, help='Size of a partition')
parser.add_argument('--partition_out', type=str, required=False, default=None, help='Output path where the partitioned prompts will be saved')

DATASET_CLASSES = {
    'MedQA': MedQA,
    'MedMCQA': MedMCQA,
    'AugmentedClinicalNotes': AugmentedClinicalNotes,
    'MedNLI': MedNLI,
    'SumPubmed': SumPubMed
}

def main():
    args = parser.parse_args()

    print('Script called with args : ', args)

    assert args.dataset in DATASET_CLASSES, f'Dataset {args.dataset} not found in list of available datasets : {DATASET_CLASSES.keys()}'

    if args.partition:
        assert args.partition_out is not None, 'Partition output path is required'

    dataset = DATASET_CLASSES[args.dataset](args.dataset_path)

    dataset.generate_prompts(output_path=args.out)
    logger.info(f'Prompts generated and saved to {args.out}')

    if args.partition:
        dataset = Dataset(args.out)
        dataset.partition(output_folder_path=args.partition_out, size_of_partition=args.size)

        logger.info(f'Prompts partitioned and saved to {args.partition_out}')

if __name__ == '__main__':
    main()
