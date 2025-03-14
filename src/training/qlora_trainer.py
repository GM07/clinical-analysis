from peft import prepare_model_for_kbit_training
import logging

from src.models.loading_config import LoadingConfig
from src.training.lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)

class QLoRATrainer(LoRATrainer):
    def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path):
        super().__init__(model_checkpoint, tokenizer_checkpoint, dataset_path)

    def get_loading_config(self) -> LoadingConfig:
        return LoadingConfig(pad_equals_eos=True, padding_side='right', use_quantization=True)
    
    def prepare_training(self):
        logger.info('Preparing model for k-bit training')
        self.model = prepare_model_for_kbit_training(self.model)

        LoRATrainer.prepare_training(self)
