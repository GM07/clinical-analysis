from dataclasses import dataclass
import logging

from tqdm import tqdm
from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.generation.ontology_prompter import OntologyPrompter, OntologyConstrainedModel
from src.generation.ontology_beam_scorer import GenerationConfig
from src.ontology.medcat_annotator import MedCatAnnotator
from src.ontology.snomed import Snomed
from src.pipelines.pipeline import Pipeline
from src.model_registry import LoadingConfig, ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class ExtractionPipelineConfig:

    batch_size: int = 5 # Number of concepts to process in parallel
    nb_concepts: int = 5 # Number of total concepts to process
    save_frequency: int = 1 # Number of results saved between partition saves

"""
Types of extraction pipelines based on the decoding method used :
- ExtractionPipeline, DatasetExtractionPipeline, PartitionedExtractionPipeline : These pipelines only run the prompter using the given generation config (only one decoding method for the input)
- ComparisonExtractionPipeline, DatasetComparisonExtractionPipeline, PartitionedComparisonExtractionPipeline : These pipelines run the prompter on all three decoding methods

Types of extraction pipelines based on the input type :
- ExtractionPipeline, ComparisonExtractionPipeline : These expect a single clinical note per call (usually for debugging)
- DatasetExtractionPipeline, DatasetComparisonExtractionPipeline : These expect a huggingface dataset and will process all clinical notes in the dataset. Clinical notes are expected in the column TEXT and Result will be added to a column OUTPUT
- PartitionedExtractionPipeline, PartitionedComparisonExtractionPipeline : These expect a dataset partition (see DatasetPartition)
"""


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
        loading_config: LoadingConfig = LoadingConfig(),
        tokenizer_path: str = None,
        system_prompt: str = None,
        apply_chat_template: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
            loading_config: Loading configuration used to load the model
            system_prompt: System prompt used to generate the prompts
        """
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path
        self.medcat_path = medcat_path
        self.medcat_device = medcat_device
        self.loading_config = loading_config
        self.system_prompt = system_prompt
        self.apply_chat_template = apply_chat_template

        self.load()

    def load(self):
        """
        Loads the model, the tokenizer, the snomed ontology and the medcat annotator
        """
        if self.tokenizer_path is None:
            self.model, self.tokenizer = ModelRegistry.load_single_checkpoint(self.checkpoint_path, loading_config=self.loading_config)
        else:
            self.model = ModelRegistry.load_single_model(self.checkpoint_path)
            self.tokenizer = ModelRegistry.load_single_tokenizer(self.tokenizer_path)
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path, nb_classes=366771)
        self.medcat = MedCatAnnotator(self.medcat_path, device=self.medcat_device)
 
        self.ontology_constrained_model = OntologyConstrainedModel(
            model=self.model,
            tokenizer=self.tokenizer,
            snomed=self.snomed,
            annotator=self.medcat,
            apply_chat_template=self.apply_chat_template
        )

    def __call__(self, clinical_note: str, generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            clinical_note: Clinical note onto which the pipeline will be ran un
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=generation_config
        )

        return results

    def get_results(clinical_note: str, prompter: OntologyPrompter, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        normal_config = GenerationConfig.greedy_search()
        beam_config = GenerationConfig.beam_search()
        constrained_config = GenerationConfig.ontology_beam_search()

        normal_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=normal_config
        )

        beam_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=beam_config
        )

        constrained_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=constrained_config
        )

        results = (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)
        return results


class DatasetExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator. Takes a dataset of clinical notes as an input and returns a dataset of extracted concepts. If greedy
    search is used, vllm will be used.
    """

    def __call__(
        self, 
        dataset: HuggingFaceDataset,
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), 
        extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()
    ):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = []
        for row in tqdm(dataset, total=len(dataset)):
            clinical_note = row['TEXT']
            attributes = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=generation_config
            )
            results.append(attributes)

        dataset = dataset.add_column('OUTPUT', results)
        return dataset

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
        loading_config: LoadingConfig = LoadingConfig(),
        tokenizer_path: str = None,
        system_prompt: str = None,
        apply_chat_template: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to the model (if path does not exist locally, the model will be fetched)
            snomed_path: Path to snomed owl file
            snomed_cache_path: Path to snomed cache file
            medcat_path: Path to medcat annotator model
            medcat_device: Device used by the medcat annotator
        """
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

    def __call__(
        self, 
        partition: DatasetPartition, 
        generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), 
        extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()
    ):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            clinical_note = value['TEXT']
            attributes = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=generation_config
            )
            results.append((i, attributes))

            if i % extraction_config.save_frequency == 0:
                partition.save_results(results)
                results = []

        partition.save_results(results)


class ComparisonExtractionPipeline(ExtractionPipeline):
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
        loading_config: LoadingConfig = LoadingConfig(),
        tokenizer_path: str = None,
        system_prompt: str = None,
        apply_chat_template: bool = True
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
        super().__init__(checkpoint_path, snomed_path, snomed_cache_path, medcat_path, medcat_device, loading_config, tokenizer_path, system_prompt, apply_chat_template)

    def __call__(self, clinical_note: str, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            clinical_note: Clinical note to run the pipeline on
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        return self.get_results(clinical_note, prompter, extraction_config)

    def get_results(clinical_note: str, prompter: OntologyPrompter, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        normal_config = GenerationConfig.greedy_search()
        beam_config = GenerationConfig.beam_search()
        constrained_config = GenerationConfig.ontology_beam_search()

        normal_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=normal_config
        )

        beam_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=beam_config
        )

        constrained_attr_by_id = prompter.start_multiple(
            clinical_notes=[clinical_note],
            top_n=extraction_config.nb_concepts,
            batch_size=extraction_config.batch_size,
            generation_config=constrained_config
        )

        results = (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)
        return results

class DatasetComparisonExtractionPipeline(ComparisonExtractionPipeline):
    """
    Same as `ComparisonExtractionPipeline` but takes a dataset as an input and returns a dataset of extracted concepts.
    It will not use vllm.
    """

    def __call__(self, dataset: HuggingFaceDataset, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = []
        for row in tqdm(dataset, total=len(dataset)):
            clinical_note = row['TEXT']
            result = ComparisonExtractionPipeline.get_results(self, clinical_note, prompter, extraction_config)
            results.append(result)

        dataset = dataset.add_column('OUTPUT', results)

        return dataset

class PartitionedComparisonExtractionPipeline(ComparisonExtractionPipeline):
    """
    Same as `ComparisonExtractionPipeline` but takes a partition as an input and returns a saves the results in the partition.
    """

    def __call__(self, partition: DatasetPartition, extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = OntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = []
        for i, value in tqdm(partition.iterate(), total=partition.nb_elements_unprocessed()):
            clinical_note = value['TEXT']
            results = ComparisonExtractionPipeline.get_results(self, clinical_note, prompter, extraction_config)

            results.append((i, results))

            if i % extraction_config.save_frequency == 0:
                partition.save_results(results)
                results = []

        partition.save_results(results)
