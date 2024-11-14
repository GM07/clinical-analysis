from dataclasses import dataclass
import logging
from typing import List

from tqdm import tqdm

from src.dataset import DatasetPartition
from src.generation.generation import OntologyBasedPrompter, OntologyConstrainedModel
from src.generation.ontology_beam_scorer import GenerationConfig
from src.ontology.annotator import MedCatAnnotator
from src.ontology.snomed import Snomed
from src.pipelines.pipeline import Pipeline
from src.model_registry import LoadingConfig, ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class ExtractionPipelineConfig:

    batch_size: int = 5
    nb_concepts: int = 5
    save_frequency: int = 1

class ExtractionPipeline(Pipeline):
    """
    Pipeline used to extract information from clinical notes using ontological concepts tagged by the MedCat
    annotator. This will output a single file containing, for each clinical notes and ontological concepts, 
    a generation made using greedy search, a generation made using diverse beam search and one made using 
    ontology-based beam search.
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
            print('normal : ', normal_attr_by_id)

            beam_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=beam_config
            )

            print('beam : ', beam_attr_by_id)

            constrained_attr_by_id, _ = prompter.start_multiple(
                clinical_notes=[clinical_note],
                top_n=extraction_config.nb_concepts,
                batch_size=extraction_config.batch_size,
                generation_config=constrained_config
            )

            print('constrained : ', constrained_attr_by_id)

            results.append((i, (normal_attr_by_id, beam_attr_by_id, constrained_attr_by_id)))

            if i % extraction_config.save_frequency == 0:
                partition.save_results(results)
                results = []

        partition.save_results(results)
