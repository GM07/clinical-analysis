from dataclasses import dataclass
import logging
from typing import List

from datasets import Dataset as HuggingFaceDataset

from src.data.dataset import DatasetPartition
from src.generation.guided_ontology_prompter import GuidedOntologyPrompter
from src.generation.ontology_constrained_model import OntologyConstrainedModel
from src.generation.ontology_beam_scorer import GenerationConfig
from src.ontology.medcat_annotator import MedCatAnnotator
from src.ontology.snomed import Snomed
from src.pipelines.pipeline import Pipeline
from src.model_registry import LoadingConfig, ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class ExtractionPipelineConfig:

    nb_concepts: int = 5 # Number of total concepts to process
    save_frequency: int = 1 # Number of results saved between partition saves
    return_internal_dataset: bool = True
    save_internal_dataset: bool = False
    internal_dataset_saving_path: str = None

@dataclass
class ComparisonExtractionPipelineConfig(ExtractionPipelineConfig):
    batch_size: int = 5

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
            system_prompt: System prompt used to generate the prompts,
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

    def __call__(self, clinical_notes: List[str], generation_config: GenerationConfig = GenerationConfig.ontology_beam_search(), extraction_config: ExtractionPipelineConfig = ExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            clinical_notes: Clinical notes onto which the pipeline will be ran on
            generation_config: Configuration for the generation
            extraction_config: Configuration for the extraction
        """

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = prompter(
            clinical_notes=clinical_notes,
            top_n=extraction_config.nb_concepts,
            generation_config=generation_config,
            return_dataset=extraction_config.return_internal_dataset or extraction_config.save_internal_dataset,
        )

        if extraction_config.save_internal_dataset:
            answers, dataset = results
            small_dataset = dataset.remove_columns(['clinical_note']) # clinical notes can be large so we don't save these
            small_dataset.to_csv(extraction_config.internal_dataset_saving_path)        

        if extraction_config.return_internal_dataset:
            answers, dataset = results
            return answers, dataset

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

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        results = prompter(
            clinical_notes=dataset['TEXT'],
            top_n=extraction_config.top_n,
            generation_config=generation_config,
            return_dataset=extraction_config.return_internal_dataset or extraction_config.save_internal_dataset
        )

        if extraction_config.save_internal_dataset:
            answers, dataset = results
            small_dataset = dataset.remove_columns(['clinical_note']) # clinical notes can be large so we don't save these
            small_dataset.to_csv(extraction_config.internal_dataset_saving_path)

        if extraction_config.return_internal_dataset:
            return answers, dataset

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

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        ids, values = partition.get_unprocessed_items(separate=True)
        notes = list(filter(lambda x: x['TEXT'], values))
        results = prompter(
            clinical_notes=notes,
            top_n=extraction_config.nb_concepts,
            generation_config=generation_config,
            return_dataset=False
        )

        saving_results = []
        for id, result in zip(ids, results):
            saving_results.append((id, result))

        partition.save_results(saving_results)

class ComparisonExtractionPipeline(ExtractionPipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator. This pipeline will return for each sample in the dataset a generation made using greedy search, 
    a generation made using diverse beam search and one made using ontology-based beam search. 
    """

    def __call__(self, clinical_notes: List[str], extraction_config: ComparisonExtractionPipelineConfig = ComparisonExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            clinical_note: Clinical note to run the pipeline on
            extraction_config: Configuration for the extraction
        """

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )

        return self.get_results(clinical_notes, prompter, extraction_config)

    def get_results(self, clinical_notes: List[str], prompter: GuidedOntologyPrompter, extraction_config: ComparisonExtractionPipelineConfig = ComparisonExtractionPipelineConfig()):
        # TODO : Support for saving internal dataset
        normal_config = GenerationConfig.greedy_search(batch_size=extraction_config.batch_size)
        beam_config = GenerationConfig.beam_search(batch_size=extraction_config.batch_size)
        constrained_config = GenerationConfig.ontology_beam_search(batch_size=extraction_config.batch_size)

        normal_attr_by_id = prompter(
            clinical_notes=clinical_notes,
            top_n=extraction_config.nb_concepts,
            generation_config=normal_config
        )

        beam_attr_by_id = prompter(
            clinical_notes=clinical_notes,
            top_n=extraction_config.nb_concepts,
            generation_config=beam_config
        )

        constrained_attr_by_id = prompter(
            clinical_notes=clinical_notes,
            top_n=extraction_config.nb_concepts,
            generation_config=constrained_config,
        )

        results = (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)
        return results


class DatasetComparisonExtractionPipeline(ComparisonExtractionPipeline):
    """
    Same as `ComparisonExtractionPipeline` but takes a dataset as an input and returns a dataset of extracted concepts.
    It will not use vllm.
    """

    def __call__(self, dataset: HuggingFaceDataset, extraction_config: ComparisonExtractionPipelineConfig = ComparisonExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            dataset: Dataset onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )


        results = ComparisonExtractionPipeline.get_results(self, dataset['TEXT'], prompter, extraction_config)
        dataset = dataset.add_column('normal', results[0])
        dataset = dataset.add_column('beam', results[1])
        dataset = dataset.add_column('constrained', results[2])

        return dataset

class PartitionedComparisonExtractionPipeline(ComparisonExtractionPipeline):
    """
    Same as `ComparisonExtractionPipeline` but takes a partition as an input and returns a saves the results in the partition.
    """

    def __call__(self, partition: DatasetPartition, extraction_config: ComparisonExtractionPipelineConfig = ComparisonExtractionPipelineConfig()):
        """
        Executes the pipeline on the dataset

        Args:
            partition: Partition onto which the pipeline will be ran
            extraction_config: Configuration for the extraction
        """

        prompter = GuidedOntologyPrompter(
            constrained_model=self.ontology_constrained_model,
            snomed=self.snomed,
            annotator=self.medcat,
            system_prompt=self.system_prompt
        )


        ids, values = partition.get_unprocessed_items()
        notes = list(map(lambda x: x['TEXT'], values))
        normal, beam, constrained = ComparisonExtractionPipeline.get_results(self, notes, prompter, extraction_config)

        saving_results = []
        for id, n, b, c in zip(ids, normal, beam, constrained):
            saving_results.append((id, (n, b, c)))

        partition.save_results(saving_results)
