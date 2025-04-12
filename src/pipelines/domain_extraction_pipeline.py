from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import List

from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.domain_ontology_prompter import DomainOntologyPrompter
from src.generation.ontology_beam_scorer import GenerationConfig
from src.pipelines.extraction_pipeline import ExtractionPipeline
from src.model_registry import LoadingConfig


logger = logging.Logger(__name__)

@dataclass
class DomainExtractionPipelineConfig:
    batch_size: int = 1
    return_internal_dataset: bool = True
    save_internal_dataset: bool = False
    internal_dataset_saving_path: str = None

class DomainExtractionPipeline(ExtractionPipeline):

    def __init__(
        self, 
        checkpoint_path: str, 
        snomed_path: str, 
        snomed_cache_path: str, 
        medcat_path: str, 
        dcf_paths: List[str],
        medcat_device: str = 'cuda', 
        loading_config: LoadingConfig = ..., 
        tokenizer_path: str = None, 
        system_prompt: str = None
    ):
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt)

        self.dcf_paths = dcf_paths

    def load_dcfs(self):
        try:
            self.concept_set = DomainClassFrequency.get_concepts_from_dcf_paths(self.dcf_paths)
        except Exception as e:
            e.add_note(f'Could not load dcf paths : {self.dcf_paths}')
            raise e

    def __call__(
        self, 
        clinical_notes: List[str],
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(batch_size=1),
        extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()
    ):     
        """
        Executes the pipeline on the dataset

        Args:
            clinical_notes: Clinical notes to run the pipeline on
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """

        if extraction_config.save_internal_dataset:
            assert extraction_config.internal_dataset_saving_path is not None, 'A path must be provided to save the internal dataset'

        prompter = DomainOntologyPrompter(
            snomed=self.snomed,
            annotator=self.medcat,
            constrained_model=self.ontology_constrained_model
        )

        result = prompter(
            clinical_notes=clinical_notes,
            domain_concept_ids=self.concept_set,
            generation_config=generation_config,
            return_dataset=extraction_config.return_internal_dataset or extraction_config.save_internal_dataset
        )

        if extraction_config.save_internal_dataset:
            answers, dataset = result
            small_dataset = dataset.remove_columns(['TEXT', 'prompt']) # clinical notes can be large so we don't save these
            small_dataset.to_csv(extraction_config.internal_dataset_saving_path)

        if extraction_config.return_internal_dataset:
            return answers, dataset

        return answers

class ComparisonDomainExtractionPipeline(DomainExtractionPipeline):

    def __call__(self, clinical_notes: List[str], extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            clinical_notes: Clinical notes to run the pipeline on
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction

        Returns:
        Tuple containing three elements : [extractions_greedy, extractions_beam, extractions_constrained]
        The extractions will be a list of the same size as `clinical_notes` and each element will contain
        a dictionary mapping the detecting 
        """
        constrained_config = GenerationConfig.ontology_beam_search(extraction_config.batch_size)
        beam_config = GenerationConfig.beam_search(extraction_config.batch_size)
        greedy_config = GenerationConfig.greedy_search(extraction_config.batch_size)

        configs = {
            'greedy': greedy_config, 
            'beam': beam_config,
            'constrained': constrained_config
        }

        results = defaultdict(list)

        for name, config in configs.items():
            logger.info(f'Trying config : ', name)
            extraction_config.return_internal_dataset = False
            extraction_config.internal_dataset_saving_path = f'{extraction_config.internal_dataset_saving_path.replace(".csv", f"{name}.csv")}' if extraction_config.internal_dataset_saving_path else None
            result = DomainExtractionPipeline.__call__(
                self,
                clinical_notes=clinical_notes,
                generation_config=config,
                extraction_config=extraction_config
            )

            results[name].append(result)

        return results
