from argparse import ArgumentParser
import logging

from src.pipelines.domain_extraction_pipeline import ComparisonDomainExtractionPipeline, DomainExtractionPipelineConfig
from src.scripts.domain_extraction_experiment import DomainExtractionExperiment

from datasets import Dataset as HuggingFaceDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Program that performs domain comparison extraction')

parser.add_argument('--experiment_folder', type=str, required=True, help='Path to experiment folder (should contain a config.yaml file)')

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)
    experiment = DomainExtractionExperiment(args.experiment_folder)
    print('Experiment settings : ', experiment.__dict__)

    dataset = HuggingFaceDataset.from_csv(experiment.dataset)
    
    pipeline = ComparisonDomainExtractionPipeline(
        checkpoint_path=experiment.checkpoint,
        tokenizer_path=experiment.tokenizer,
        dcf_paths=experiment.dcfs,
        snomed_path=experiment.snomed,
        snomed_cache_path=experiment.snomed_cache,
        medcat_path=experiment.medcat,
        medcat_device=experiment.medcat_device,
        system_prompt=experiment.system_prompt,
    )

    results = pipeline(
        clinical_notes=dataset[experiment.dataset_input_column], 
        extraction_config=DomainExtractionPipelineConfig(
            batch_size=experiment.batch_size,
            internal_dataset_saving_path=experiment.internal_dataset_saving_path,
            save_internal_dataset=True,
            return_internal_dataset=False,
        )
    )

    HuggingFaceDataset.from_dict(results).to_csv(experiment.final_results_path)


if __name__ == '__main__':
    main()
