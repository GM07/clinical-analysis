

from argparse import ArgumentParser
import logging

from datasets import load_from_disk

from src.data.dataset import DatasetPartitionAnalyzer
from src.pipelines.model_inference_pipeline import ClassifierModelInferencePipeline
from src.models.halloumi import HallOumi

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Performs the evaluation of the HallOumi model on the MedHal dataset')

parser.add_argument('--model', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--out', required=True, type=str)

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    df = load_from_disk(args.dataset)
    df = df['test']
    pipeline = ClassifierModelInferencePipeline(model_path=args.model, tokenizer_path=args.model)

    def to_prompt(x):
        return {'text': HallOumi.create_prompt_classifier(x['context'], x['statement'])}

    df = df.map(to_prompt)
    print(df)
    results = pipeline.run_inference(df['text'], apply_chat_template=False)
    df = df.add_column('results_hall', results)
    df.to_csv(args.out)

if __name__ == '__main__':
    main()
