from collections import defaultdict
from dataclasses import dataclass
import logging
import os
from typing import List

from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.domain_ontology_prompter import DomainOntologyPrompter
from src.generation.ontology_beam_scorer import GenerationConfig
from src.pipelines.extraction_pipeline import ExtractionPipeline
from src.model_registry import LoadingConfig


logger = logging.getLogger(__name__)

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
        system_prompt: str = None,
        apply_chat_template: bool = True,
    ):
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

        self.dcf_paths = dcf_paths

        self.load_dcfs()

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

        if generation_config.batch_size != extraction_config:
            logger.info(f"Mismatch between generation's batch size and extraction's batch size, updating generation to {extraction_config.batch_size}")
            generation_config.batch_size = extraction_config.batch_size
        
        result = prompter(
            clinical_notes=clinical_notes,
            domain_concept_ids=self.concept_set,
            generation_config=generation_config,
            return_dataset=extraction_config.return_internal_dataset or extraction_config.save_internal_dataset
        )

        if extraction_config.save_internal_dataset:
            answers, dataset = result
            small_dataset = dataset.remove_columns(['clinical_note']) # clinical notes can be large so we don't save these
            small_dataset.to_csv(extraction_config.internal_dataset_saving_path)

        if extraction_config.return_internal_dataset:
            return answers, dataset

        return result


class PartitionedDomainExtractionPipeline(DomainExtractionPipeline):

    def __call__(
        self, 
        partition: DatasetPartition,
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(batch_size=1),
        extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()
    ):
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

        ids, values = partition.get_unprocessed_items()
        notes = list(map(lambda x: x['TEXT'], values))
        logger.info(f'Processing {len(notes)} clinical notes in this partition')
        results = DomainExtractionPipeline.__call__(
            self, 
            clinical_notes=notes, 
            generation_config=generation_config,
            extraction_config=extraction_config
        )

        saving_results = []
        for id, res in zip(ids, results):
            saving_results.append((id, res))

        partition.save_results(saving_results)



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

        greedy = DomainExtractionPipeline.__call__(
            self,
            clinical_notes=clinical_notes,
            generation_config=greedy_config,
            extraction_config=extraction_config
        )

        beam = DomainExtractionPipeline.__call__(
            self,
            clinical_notes=clinical_notes,
            generation_config=beam_config,
            extraction_config=extraction_config
        )

        constrained = DomainExtractionPipeline.__call__(
            self,
            clinical_notes=clinical_notes,
            generation_config=constrained_config,
            extraction_config=extraction_config
        )

        return (greedy, beam, constrained)


class PartitionedComparisonDomainExtractionPipeline(DomainExtractionPipeline):

    def __call__(self, partition: DatasetPartition, extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()):
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

        ids, values = partition.get_unprocessed_items()
        notes = list(map(lambda x: x['TEXT'], values))
        logger.info(f'Processing {len(notes)} clinical notes in this partition')
        normal, beam, constrained = ComparisonDomainExtractionPipeline.__call__(self, notes, extraction_config)

        saving_results = []
        for id, n, b, c in zip(ids, normal, beam, constrained):
            saving_results.append((id, (n, b, c)))

        partition.save_results(saving_results)

