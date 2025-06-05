from argparse import ArgumentParser
import logging

from src.data.dataset import DatasetPartition
from src.generation.ontology_beam_scorer import GenerationConfig
from src.pipelines.domain_extraction_pipeline import DomainExtractionPipelineConfig, PartitionedDomainExtractionPipeline
from src.scripts.domain_extraction_experiment_config import DomainExtractionExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that performs domain comparison extraction')

parser.add_argument('--experiment_folder', type=str, required=True, help='Path to experiment folder (should contain a config.yaml file)')

methods = {
    'greedy': GenerationConfig.greedy_search(),
    'beam': GenerationConfig.beam_search(),
    'constrained': GenerationConfig.ontology_beam_search()
}

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)
    experiment = DomainExtractionExperimentConfig(args.experiment_folder)
    print('Experiment settings : ', experiment.__dict__)
    method, partition_path = experiment.infer()
    if method is None:
        print('All partitions have been processed')

    gen_config = methods[method]

    experiment.lock_partition(partition_path)
    print('Processing partition at', partition_path)

    try:
        partition = DatasetPartition.from_save(partition_path, original_dataset_path=experiment.dataset)
        
        pipeline = PartitionedDomainExtractionPipeline(
            checkpoint_path=experiment.checkpoint,
            tokenizer_path=experiment.tokenizer,
            dcf_paths=experiment.dcfs,
            snomed_path=experiment.snomed,
            snomed_cache_path=experiment.snomed_cache,
            medcat_path=experiment.medcat,
            medcat_device=experiment.medcat_device,
            system_prompt=experiment.system_prompt,
        )

        pipeline(
            partition=partition,
            generation_config=gen_config,
            extraction_config=DomainExtractionPipelineConfig(
                batch_size=experiment.batch_size,
                save_internal_dataset=False,
                return_internal_dataset=False,
            )
        )
        
    except Exception as e:
        print(e)
    finally:
        experiment.unlock_partition(partition_path)

if __name__ == '__main__':
    main()
