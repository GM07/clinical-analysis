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
parser.add_argument('--resume_from_checkpoint', type=bool, required=False, help='If True, resume from checkpoint')
parser.add_argument('--batch_size', type=int, required=False, help='Batch size used for training')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    trainer = EvaluatorTrainer(
        model_checkpoint=args.model_checkpoint,
        dataset_dict_path=args.dataset_path,
        local=False
    )


    trainer.train(batch_size=args.batch_size, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
