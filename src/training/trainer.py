from abc import abstractmethod
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

import logging

from src.models.loading_config import LoadingConfig
from src.models.utils import load_model, load_tokenizer
from src.data.formatter import Formatter

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.dataset_path = dataset_path
        self.loading_config = self.get_loading_config()

    def load_dataset(self):
        logger.info(f'Loading dataset from {self.dataset_path}')
        self.dataset = load_from_disk(self.dataset_path)

    def load_checkpoint(self):
        self.model = load_model(self.model_checkpoint, self.loading_config)
        self.tokenizer = load_tokenizer(self.tokenizer_checkpoint, self.loading_config)

    def load(self):
        self.load_dataset()
        self.load_checkpoint()

    @abstractmethod
    def prepare_training(self):
        pass

    @abstractmethod
    def train(self, output_dir: str, batch_size: int = 1, resume_from_checkpoint: str = None):

        assert (
            self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
        ), "The tokenizer's pad token id and eos token id should not be the same."

        self.prepare_training()
        
        sft_config = self.get_sft_config()
        sft_config.output_dir = output_dir
        sft_config.per_device_train_batch_size = batch_size
        sft_config.per_device_eval_batch_size = batch_size

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset['train'].select(range(2)),
            eval_dataset=self.dataset['train'].select(range(2)),
            tokenizer=self.tokenizer,
            formatting_func=Formatter(),
            args=sft_config,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_model(f'{output_dir}/model')
        self.tokenizer.save_pretrained(f'{output_dir}/model')

        self.post_training(output_dir)

    @abstractmethod
    def post_training(self, output_dir: str):
        pass

    @abstractmethod
    def get_sft_config(self) -> SFTConfig:
        pass

    @abstractmethod
    def get_loading_config(self) -> LoadingConfig:
        pass
