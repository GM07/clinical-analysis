from argparse import ArgumentParser
import logging

from src.dataset import Dataset, DatasetPartition
from src.pipelines.prometheus_pipeline import PrometheusEvaluationPipeline

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that evaluates extractions using Prometheus')

parser.add_argument('--partition', type=str, required=True, help='Path to partition file')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    pipeline = PrometheusEvaluationPipeline(
        checkpoint_path=args.checkpoint,
    )
    pipeline.load()

    partition = DatasetPartition.from_save(args.partition)

    pipeline(partition)

if __name__ == '__main__':
    main()
