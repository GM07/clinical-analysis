from argparse import ArgumentParser
import logging

from src.domain_adaptation.evaluator_trainer import EvaluatorTrainer


logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--dataset_path', type=str, required=True, help='Path to huggingface dataset used to train the evaluator')
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')

def main():

    args = parser.parse_args()


    trainer = EvaluatorTrainer(
        model_checkpoint=args.model_checkpoint,
        dataset_dict_path=args.dataset_path,
        local=False
    )

    print('Called with arguments : ', args)
    trainer = EvaluatorTrainer(
        dataset_path=args.dataset_path,
        model_checkpoint=args.model_checkpoint
    )

    trainer.train(batch_size=128)


if __name__ == '__main__':
    main()
