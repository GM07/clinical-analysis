from src.models.utils import load_model
from peft import LoraConfig, PeftModel, get_peft_model
import logging

from src.training.full_finetune_trainer import FullFinetuneTrainer

logger = logging.getLogger(__name__)

class LoRATrainer(FullFinetuneTrainer):

    def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path):
        super().__init__(model_checkpoint, tokenizer_checkpoint, dataset_path)

        self.lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            task_type='CAUSAL_LM',
        )

    def prepare_training(self):
        logger.info('Adding LoRA adapters to model')
        self.model = get_peft_model(self.model, self.lora_config)
        logger.info(f'Model has {self.model.num_parameters()} trainable parameters')

    def post_training(self, output_dir: str):
        logger.info('Merging LoRA adapters with base model')
        
        base_model = load_model(self.model_checkpoint, self.loading_config)
        peft_model = PeftModel.from_pretrained(base_model, f'{output_dir}/model')
        merged_model = peft_model.merge_and_unload()

        logger.info('Saving merged model')
        merged_model.save_pretrained(f'{output_dir}/merged_model')
        self.tokenizer.save_pretrained(f'{output_dir}/merged_model')
