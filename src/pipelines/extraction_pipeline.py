from dataclasses import dataclass
import logging
from typing import Dict, List
import sys

from tqdm import tqdm
from vllm import LLM
from datasets import Dataset as HuggingFaceDataset


from src.data.dataset import DatasetPartition
from src.generation.generation import OntologyBasedPrompter, OntologyConstrainedModel
from src.generation.ontology_beam_scorer import GenerationConfig
from src.ontology.medcat_annotator import MedCatAnnotator
from src.ontology.snomed import Snomed
from src.pipelines.dataset_inference_pipeline import HuggingFaceDatasetInferencePipeline
from src.pipelines.pipeline import Pipeline
from src.model_registry import LoadingConfig, ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class ExtractionPipelineConfig:

    batch_size: int = 5
    nb_concepts: int = 5
    save_frequency: int = 1

@dataclass
class DomainExtractionPipelineConfig(ExtractionPipelineConfig):
    nb_concepts: int = 30

class ExtractionPipeline(Pipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
            loading_config: Loading configuration used to load the model
        """
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path
        self.medcat_path = medcat_path
        self.medcat_device = medcat_device
        self.loading_config = loading_config

    def load(self):
        """
        Loads the model, the tokenizer, the snomed ontology and the medcat annotator
        """
        self.model, self.tokenizer = ModelRegistry.load_single_checkpoint(self.checkpoint_path, loading_config=self.loading_config)
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path, nb_classes=366771)
        self.medcat = MedCatAnnotator(self.medcat_path, device=self.medcat_device)
        self.ontology_constrained_model = OntologyConstrainedModel(
            model=self.model,
            tokenizer=self.tokenizer,
            snomed=self.snomed,
            annotator=self.medcat,
            apply_chat_template=True
        )
 
class DatasetExtractionPipeline(ExtractionPipeline):
    """
    Differs with the `PartitionedExtractionPipeline` in that it will store the prompts in a dataset instead of 
    the results. The dataset will then be sent to the model to be extracted using vllm. This only supports greedy search.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config)

    def load(self):
        """
        Loads the model, the tokenizer, the snomed ontology and the medcat annotator
        """
        logger.info(f'Loading snomed ontology from {self.snomed_path}')
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path, nb_classes=366771)
        logger.info(f'Loading medcat annotator from {self.medcat_path}')
        self.medcat = MedCatAnnotator(self.medcat_path, device=self.medcat_device)

        logger.info(f'Loading model from {self.checkpoint_path}')
        self.pipeline = HuggingFaceDatasetInferencePipeline(
            model_path=self.checkpoint_path,
            input_column='PROMPT',
            output_column='OUTPUT'
        )

        self.cache_dataset = None

    def __call__(
        self, 
        dataset: HuggingFaceDataset,
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), 
        extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()
    ):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """
        if generation_config == GenerationConfig.greedy_search():
            # vLLM is faster in this case
            return self.greedy_search(dataset, extraction_config)

        prompter = OntologyBasedPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
        )

        results = []
        for row in tqdm(dataset, total=len(dataset)):
            clinical_note = row['TEXT']
            constrained_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=generation_config
            )
            results.append(constrained_attr_by_id)

        dataset = dataset.add_column('OUTPUT', results)

        return dataset

    def greedy_search(self, dataset: HuggingFaceDataset, extraction_config: DomainExtractionPipelineConfig):
        if self.cache_dataset is None:
            logger.info(f'Generating temporary dataset for pipeline')
            prompter = OntologyBasedPrompter(
                snomed=self.snomed,
                annotator=self.medcat,
                dataset_mode=True
            )

            # Mapping from id of clinical note to list of prompts associated to each concept detected in the clinical note
            prompts: Dict[str, List[str]] = prompter.start_dataset(
                clinical_notes=dataset['TEXT'],
                ids=dataset['ROW_ID'],
                top_n=extraction_config.nb_concepts
            )
            
            dataset_dict = {
                'ROWID': [k for k, v in prompts.items() for _ in v],
                'PROMPT': [self.prompt_to_chat(p) for k, v in prompts.items() for p in v]
            }
            self.cache_dataset = HuggingFaceDataset.from_dict(dataset_dict)
        else:
            logger.info(f'Using cached dataset')

        logger.info(f'Running inference pipeline')
        dataset = self.pipeline(self.cache_dataset)

        return dataset
    def prompt_to_chat(self, prompt: str):
        return [{
            'role': 'user',
            'content': prompt
        }]

class DatasetComparisonExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator. This pipeline will return for each sample in the dataset a generation made using greedy search, 
    a generation made using diverse beam search and one made using ontology-based beam search. 
    """
    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
            loading_config: Loading configuration used to load the model
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config)


    def __call__(self, dataset: HuggingFaceDataset, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyBasedPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
        )

        normal_config = GenerationConfig.greedy_search()
        beam_config = GenerationConfig.beam_search()
        constrained_config = GenerationConfig.ontology_beam_search()

        results = []
        for row in tqdm(dataset, total=len(dataset)):
            clinical_note = row['TEXT']
            normal_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=normal_config
            )

            beam_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=beam_config
            )

            constrained_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=constrained_config
            )

            self.results = (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)
            results.append(constrained_attr_by_id)

        dataset = dataset.add_column('OUTPUT', results)

        return dataset

class ComparisonExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator. This pipeline will return a generation made using greedy search, a generation made using 
    diverse beam search and one made using ontology-based beam search. Only one clinical note at a time is
    supported.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
            loading_config: Loading configuration used to load the model
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config)

    def __call__(self, clinical_note: str, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyBasedPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
        )

        normal_config = GenerationConfig.greedy_search()
        beam_config = GenerationConfig.beam_search()
        constrained_config = GenerationConfig.ontology_beam_search()

        normal_attr_by_id, _ = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=normal_config
        )

        beam_attr_by_id, _ = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=beam_config
        )

        constrained_attr_by_id, _ = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=constrained_config
        )

        self.results = (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)
        return self.results

class PartitionedComparisonExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from partitioned clinical notes using ontological concepts tagged by the MedCat
    annotator. This will save a partition containing, for each clinical notes and ontological concepts, 
    a generation made using greedy search, a generation made using diverse beam search and one made using 
    ontology-based beam search. This pipeline is mainly used to compare decoding methods in the extraction
    phase.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
            loading_config: Loading configuration used to load the model
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config)

    def __call__(self, partition: DatasetPartition, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyBasedPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
        )

        normal_config = GenerationConfig.greedy_search()
        beam_config = GenerationConfig.beam_search()
        constrained_config = GenerationConfig.ontology_beam_search()

        results = []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            clinical_note = value['TEXT']
            normal_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=normal_config
            )
            # print('normal : ', normal_attr_by_id)

            beam_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=beam_config
            )

            # print('beam : ', beam_attr_by_id)

            constrained_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=constrained_config
            )

            # print('constrained : ', constrained_attr_by_id)

            results.append((i, (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)))

            if i % extraction_config.save_frequency == 0:
                partition.save_results(results)
                results = []

        partition.save_results(results)

class PartitionedExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from partitioned clinical notes using ontological concepts tagged by the MedCat
    annotator. This will store into the `result` attribute, for each clinical notes and ontological concepts, 
    a generation made using greedy search, a generation made using diverse beam search and one made using 
    ontology-based beam search. The difference with the `PartitionedComparisonExtractionPipeline` is that the `PartitionedExtractionPipeline`
    will only extract the concepts using a specific decoding method.
    """

    def __init__(
        self, 
        checkpoint_path: str,
        snomed_path: str,
        snomed_cache_path: str,
        medcat_path: str,
        medcat_device: str = 'cuda',
        loading_config: LoadingConfig = LoadingConfig()
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config)

    def __call__(
        self, 
        partition: DatasetPartition, 
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), 
        extraction_config: DomainExtractionPipelineConfig = DomainExtractionPipelineConfig()
    ):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyBasedPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
        )

        results = []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            clinical_note = value['TEXT']
            constrained_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=generation_config
            )
            results.append((i, constrained_attr_by_id))

            if i % extraction_config.save_frequency == 0:
                partition.save_results(results)
                results = []

        partition.save_results(results)
