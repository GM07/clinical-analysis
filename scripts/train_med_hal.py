from argparse import ArgumentParser
import logging

from src.training.med_hal_trainer import MedHALTrainer
from src.model_registry import LoadingConfig

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that trains a MedHAL model')

parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--tokenizer_checkpoint', type=str, help='Path to tokenizer checkpoint')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    tokenizer_checkpoint = args.tokenizer_checkpoint if args.tokenizer_checkpoint else args.model_checkpoint

    trainer = MedHALTrainer(
        model_checkpoint=args.model_checkpoint, 
        tokenizer_checkpoint=tokenizer_checkpoint,
        dataset_path=args.dataset_path
    )

    loading_config = LoadingConfig(pad_equals_eos=True, padding_side='right', use_quantization=True)

    trainer.load(loading_config)
    trainer.train(args.output_dir)


if __name__ == '__main__':
    main()
