import joblib

from dataclasses import dataclass
from typing import Dict, List, Set
from datasets import Dataset
import logging
import os
import copy


from src.generation.domain_ontology_prompter import DomainOntologyPrompter
from src.generation.guided_ontology_prompter import GuidedOntologyPrompter
from src.generation.ontology_beam_scorer import GenerationConfig
from src.models.loading_config import LoadingConfig
from src.pipelines.domain_extraction_pipeline import DomainExtractionPipeline
from src.pipelines.extraction_pipeline import ComparisonExtractionPipeline, ComparisonExtractionPipelineConfig


logger = logging.getLogger(__name__)

@dataclass
class AblationPipelineConfig(ComparisonExtractionPipelineConfig):
    save_every_config: bool = True # If True, will save the config results everytime it is done processing.
    saving_path: str = None # Must contain {config_name} in the path
    generation_configs: Set[str] = None

class AblationPipeline(ComparisonExtractionPipeline):

    @staticmethod
    def create_default_configs():
        return {
            'hps': GenerationConfig.ontology_beam_search(h_score=1.0, p_score=1.0, s_score=1.0),
            'hp': GenerationConfig.ontology_beam_search(h_score=1.0, p_score=1.0, s_score=0.0),
            'hs': GenerationConfig.ontology_beam_search(h_score=1.0, p_score=0.0, s_score=1.0),
            'ps': GenerationConfig.ontology_beam_search(h_score=0.0, p_score=1.0, s_score=1.0),
            'h': GenerationConfig.ontology_beam_search(h_score=1.0, p_score=0.0, s_score=0.0),
            'p': GenerationConfig.ontology_beam_search(h_score=0.0, p_score=1.0, s_score=0.0),
            's': GenerationConfig.ontology_beam_search(h_score=0.0, p_score=0.0, s_score=1.0),
            'pn': GenerationConfig.ontology_beam_search(h_score=0.0, p_score=1.0, s_score=0.0, use_rouge_for_restrictions=False),
            'beam': GenerationConfig.beam_search(),
            'normal': GenerationConfig.greedy_search(),
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
    ):
        ComparisonExtractionPipeline.__init__(self, checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

        self.configs = AblationPipeline.create_default_configs()

    def get_results(self, clinical_notes: List[str], prompter: GuidedOntologyPrompter, extraction_config: AblationPipelineConfig = AblationPipelineConfig()):
        results = {}

        # Verify if saving is needed and that everything that paths are valid for saving results
        should_save = False
        if extraction_config.save_every_config:
            should_save = True
            assert extraction_config.saving_path is not None, 'Cannot saving ablation after each configuration without a saving path.'
            assert '{config_name}' in extraction_config.saving_path, 'Saving path must contain the string {config_name}'

        # Configurations that will be ran
        running_configs = self.verify_configs(extraction_config.generation_configs)
        
        dataset_cache: Dataset = None
        
        for name, config in self.configs.items():
            
            if name not in running_configs:
                continue

            if should_save:
                config_saving_path = extraction_config.saving_path.replace('{config_name}', name)

            if should_save and os.path.exists(config_saving_path):
                # Config was already ran
                logger.info(f'Config {name} was already ran, loading it from {config_saving_path}')
                config_results = joblib.load(config_saving_path)
            else:
                logger.info(f'Processing config {name} -> Boost factors : {config.score_boost_factors}')
                config.batch_size = extraction_config.batch_size

                if should_save:
                    prompter.log_path = extraction_config.saving_path.replace('{config_name}', f'{name}_logs')
                
                output = prompter(
                    clinical_notes=clinical_notes,
                    top_n=extraction_config.nb_concepts,
                    generation_config=copy.deepcopy(config),
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
            
            if should_save:
                joblib.dump(config_results, config_saving_path)
            
            results[name] = config_results

        return results


    def verify_configs(self, configs: Set[str]):
        if configs is None:
            return set(AblationPipeline.create_default_configs().keys())
        
        valid_configs = set(AblationPipeline.create_default_configs().keys())
        for config in configs:
            assert config in valid_configs, f'Config {config} is not a valid config. Valid configs are {valid_configs}'

        return configs


class AblationDomainPipeline(DomainExtractionPipeline):

    def __init__(
        self, 
        checkpoint_path: str, 
        snomed_path: str, 
        snomed_cache_path: str, 
        medcat_path: str, 
        dcf_paths: List[str],
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig(), 
        tokenizer_path: str = None, 
        system_prompt: str = None, 
        apply_chat_template: bool = True, 
    ):
        DomainExtractionPipeline.__init__(self, checkpoint_path, snomed_path, snomed_cache_path, medcat_path, dcf_paths, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

        self.configs = AblationPipeline.create_default_configs()

    def __call__(
        self, 
        clinical_notes: List[str],
        extraction_config: AblationPipelineConfig = AblationPipelineConfig()
    ):     
        """
        Executes the ablation study on the dataset

        Args:
            clinical_notes: Clinical notes to run the ablation study on
            extraction_config: Configuration for the extraction
        """

        if extraction_config.save_internal_dataset:
            assert extraction_config.internal_dataset_saving_path is not None, 'A path must be provided to save the internal dataset'

        prompter = DomainOntologyPrompter(
            snomed=self.snomed,
            annotator=self.medcat,
            constrained_model=self.ontology_constrained_model
        )

        return self.get_results(clinical_notes, prompter, extraction_config)


    def get_results(self, clinical_notes: List[str], prompter: DomainOntologyPrompter, extraction_config: AblationPipelineConfig = AblationPipelineConfig()):
        results = {}
        if extraction_config.save_every_config:
            assert extraction_config.saving_path is not None, 'Cannot saving ablation after each configuration without a saving path.'
            assert '{config_name}' in extraction_config.saving_path, 'Saving path must contain the string {config_name}'

        generation_configs = self.verify_configs(extraction_config.generation_configs)

        dataset_cache: Dataset = None
        for name, config in self.configs.items():
            if name not in generation_configs:
                continue

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
                    domain_concept_ids=self.concept_set,
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


    def verify_configs(self, configs: Set[str]):
        if configs is None:
            return set(AblationPipeline.create_default_configs.keys())
        
        valid_configs = set(AblationPipeline.create_default_configs.keys())
        for config in configs:
            assert config in valid_configs, f'Config {config} is not a valid config. Valid configs are {valid_configs}'

        return configs
