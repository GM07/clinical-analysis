from argparse import ArgumentParser
import logging

from datasets import DatasetDict

from src.data.validation.medhal_validator import MedHalValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description='Program that validates the samples of MedHal using a model')
parser.add_argument('--dataset', type=str, required=True, help='Path to MedHal dataset')
parser.add_argument('--model', type=str, required=True, help='Path to validator model')
parser.add_argument('--out', type=str, required=True, help='Path where to save the dataset')


def main():
    args = parser.parse_args()

    print('Script called with args : ', args)

    validator = MedHalValidator(
        medhal_path=args.dataset,
        model_path=args.model,
    )

    out: DatasetDict = validator.validate()
    out.save_to_disk(args.out)

if __name__ == '__main__':
    main()
