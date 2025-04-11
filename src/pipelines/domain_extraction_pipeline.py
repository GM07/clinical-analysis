from dataclasses import dataclass
import logging
import os
from typing import Dict, List
import sys

from tqdm import tqdm
from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.domain_adaptation.domain_class_frequency import DomainClassFrequency
from src.generation.generation import OntologyBasedPrompter, OntologyConstrainedModel, DomainOntologyPrompter
from src.generation.ontology_beam_scorer import GenerationConfig
from src.ontology.medcat_annotator import MedCatAnnotator
from src.ontology.snomed import Snomed
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline
from src.pipelines.extraction_pipeline import ExtractionPipeline
from src.pipelines.pipeline import Pipeline
from src.model_registry import LoadingConfig, ModelRegistry


@dataclass
class DomainExtractionPipelineConfig:
    batch_size: int = 1
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
            dataset: Dataset onto which the pipeline will be ran
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
            return_dataset=extraction_config.save_internal_dataset
        )
