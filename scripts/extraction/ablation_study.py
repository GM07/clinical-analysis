from argparse import ArgumentParser
import logging

from src.pipelines.ablation_pipeline import AblationPipeline, AblationPipelineConfig

from datasets import Dataset as HuggingFaceDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that extracts information using ontology-based constrained decoding')

parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset csv file')
parser.add_argument('--output_path', type=str, required=True, help='Path to output dataset csv file')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--tokenizer', type=str, default=None, help='Model tokenizer')
parser.add_argument('--snomed', type=str, help='Path to snomed ontology file (.owx)')
parser.add_argument('--snomed_cache', type=str, help='Path to snomed cache file')
parser.add_argument('--medcat', type=str, help='Path to medcat annotator checkpoint')
parser.add_argument('--nb_concepts', type=int, default=5, help='Number of concepts to extract')
parser.add_argument('--batch_size', type=int, default=5, help='Number of examples to process in parallel')
parser.add_argument('--configs', type=str, default='all', help='Which config to use separated with a comma (hps, hp, hs, ps, h, p, s, beam, normal)')

CONFIG_VALUES = ['hps', 'hp', 'hs', 'ps', 'h', 'p', 'pn', 's', 'beam', 'normal', 'all']

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)


    configs = set(map(lambda x: x.strip(), args.configs.split(',')))

    for config in configs:
        assert config in CONFIG_VALUES, f'Config {config} is not valid'

    print('Loading dataset from : ', args.dataset_path)
    dataset = HuggingFaceDataset.from_csv(args.dataset_path)

    pipeline = AblationPipeline(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        snomed_path=args.snomed,
        snomed_cache_path=args.snomed_cache,
        medcat_path=args.medcat,
        medcat_device='cuda',
    )

    extraction_config = AblationPipelineConfig(
        nb_concepts=args.nb_concepts, 
        batch_size=int(args.batch_size),
        save_every_config=True,
        saving_path=args.output_path.replace('.csv', '_{config_name}.joblib'),
        generation_configs=configs
    )

    output_results = pipeline(dataset['TEXT'], extraction_config)
    output_dataset = HuggingFaceDataset.from_dict(output_results)
    
    print('Saving dataset to : ', args.output_path)
    output_dataset.to_csv(args.output_path)

if __name__ == '__main__':
    main()
