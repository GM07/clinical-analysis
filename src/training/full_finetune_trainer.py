from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import logging
import torch

from src.models.loading_config import LoadingConfig
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

class FullFinetuneTrainer(Trainer):
    def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path):
        super().__init__(model_checkpoint, tokenizer_checkpoint, dataset_path)

    def get_sft_config(self):
        return SFTConfig(
            output_dir='output',
            gradient_accumulation_steps=16,  # Increased for full fine-tuning
            gradient_checkpointing=True,
            max_steps=1000,
            lr_scheduler_type='cosine',
            learning_rate=1e-5,  # Lower learning rate for full fine-tuning
            warmup_steps=100,
            optim="adamw_torch",
            save_strategy='steps',
            save_steps=100,
            evaluation_strategy='steps',
            eval_steps=100,
            bf16=True,
            seed=42,
            max_seq_length=8192,
        )

    def get_loading_config(self) -> LoadingConfig:
        return LoadingConfig(pad_equals_eos=True, padding_side='right')

    def post_training(self, output_dir: str):
        return
    
    def prepare_training(self):
        return
    