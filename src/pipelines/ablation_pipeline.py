import joblib

from dataclasses import dataclass
from typing import Dict, List
from datasets import Dataset
import logging
import os


from src.generation.guided_ontology_prompter import GuidedOntologyPrompter
from src.generation.ontology_beam_scorer import GenerationConfig
from src.models.loading_config import LoadingConfig
from src.pipelines.extraction_pipeline import ComparisonExtractionPipeline, ComparisonExtractionPipelineConfig


logger = logging.getLogger(__name__)

@dataclass
class AblationPipelineConfig(ComparisonExtractionPipelineConfig):
    save_every_config: bool = True # If True, will save the config results everytime it is done processing.
    saving_path: str = None # Must contain {config_name} in the path

class AblationPipeline(ComparisonExtractionPipeline):

    DEFAULT_CONFIGS = {
        # 'hps': GenerationConfig.ontology_beam_search().with_hierarchy_score(3.0).with_property_score(1.0).with_similarity_score(10.0),
        'ps': GenerationConfig.ontology_beam_search().with_hierarchy_score(0.0).with_property_score(1.0).with_similarity_score(10.0),
        'hs': GenerationConfig.ontology_beam_search().with_hierarchy_score(3.0).with_property_score(0.0).with_similarity_score(10.0),
        'hp': GenerationConfig.ontology_beam_search().with_hierarchy_score(3.0).with_property_score(1.0).with_similarity_score(0.0),
        'h': GenerationConfig.ontology_beam_search().with_hierarchy_score(3.0).with_property_score(0.0).with_similarity_score(0.0),
        'p': GenerationConfig.ontology_beam_search().with_hierarchy_score(0.0).with_property_score(1.0).with_similarity_score(0.0),
        's': GenerationConfig.ontology_beam_search().with_hierarchy_score(0.0).with_property_score(0.0).with_similarity_score(10.0),
    }

    def __init__(
        self, 
        checkpoint_path: str, 
        snomed_path: str, 
        snomed_cache_path: str, 
        medcat_path: str, 
        medcat_device: str = 'cuda', 
        loading_config: LoadingConfig = LoadingConfig(), 
        tokenizer_path: str = None, 
        system_prompt: str = None, 
        apply_chat_template: bool = True, 
        configs: Dict[str, GenerationConfig] = None
    ):
        ComparisonExtractionPipeline.__init__(self, checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

        if configs is None:
            configs = AblationPipeline.DEFAULT_CONFIGS
        self.configs = configs

    def get_results(self, clinical_notes: List[str], prompter: GuidedOntologyPrompter, extraction_config: AblationPipelineConfig = AblationPipelineConfig()):
        results = {}
        if extraction_config.save_every_config:
            assert extraction_config.saving_path is not None, 'Cannot saving ablation after each configuration without a saving path.'
            assert '{config_name}' in extraction_config.saving_path, 'Saving path must contain the string {config_name}'

        dataset_cache: Dataset = None
        for name, config in self.configs.items():
            config_saving_path = extraction_config.saving_path.replace('{config_name}', name)
            if extraction_config.save_every_config and os.path.exists(config_saving_path):
                # Config was already ran
                logger.info(f'Config {name} was already ran, loading it from {config_saving_path}')
                config_results = joblib.load(config_saving_path)
            else:
                logger.info(f'Processing config {name}')
                config.batch_size = extraction_config.batch_size
                output = prompter(
                    clinical_notes=clinical_notes,
                    top_n=extraction_config.nb_concepts,
                    generation_config=config,
                    return_dataset=dataset_cache is None, # Only return it the first time
                    dataset_cache=dataset_cache
                )

                if dataset_cache is None:
                    config_results, dataset_cache = output
                else:
                    config_results = output
                
                # We don't want to keep the results for the next iteration
                if 'result' in dataset_cache.column_names:
                    dataset_cache = dataset_cache.remove_columns('result') 
            
            if extraction_config.save_every_config:
                joblib.dump(config_results, config_saving_path)
            
            results[name] = config_results

        return results


# class DomainAblationPipeline(DomainExtractionPipeline):

#     def __call__(self, clinical_notes: List[str], extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()):
#         """
#         Executes the pipeline on the dataset

#         Args:
#             clinical_notes: Clinical notes to run the pipeline on
#             extraction_config: Configuration for the extraction
#         """

#         if extraction_config.save_internal_dataset:
#             assert extraction_config.internal_dataset_saving_path is not None, 'A path must be provided to save the internal dataset'

#         prompter = DomainOntologyPrompter(
#             snomed=self.snomed,
#             annotator=self.medcat,
#             constrained_model=self.ontology_constrained_model
#         )

#         if generation_config.batch_size != extraction_config:
#             logger.info(f"Mismatch between generation's batch size and extraction's batch size, updating generation to {extraction_config.batch_size}")
#             generation_config.batch_size = extraction_config.batch_size
        
#         result = prompter(
#             clinical_notes=clinical_notes,
#             domain_concept_ids=self.concept_set,
#             generation_config=generation_config,
#             return_dataset=extraction_config.return_internal_dataset or extraction_config.save_internal_dataset
#         )

#         return result
