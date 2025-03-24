import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import logging

from src.data.formatter import Formatter
from src.training.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)

class UMedHalTrainer:

    def __init__(self, trainer_config: TrainerConfig):
        self.trainer_config = trainer_config
        self.load_checkpoint()

    def load_checkpoint(self):
        logger.info("Loading checkpoint")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.trainer_config.checkpoint_config.model_checkpoint,
            max_seq_length=self.trainer_config.checkpoint_config.max_seq_len,
            dtype=self.trainer_config.checkpoint_config.dtype,
            load_in_4bit=self.trainer_config.checkpoint_config.load_in_4bit,
            load_in_8bit=self.trainer_config.checkpoint_config.load_in_8bit,
            trust_remote_code=self.trainer_config.checkpoint_config.trust_remote_code,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = self.trainer_config.checkpoint_config.r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] ,
            lora_alpha=self.trainer_config.checkpoint_config.lora_alpha,
            lora_dropout=self.trainer_config.checkpoint_config.lora_dropout,
            bias=self.trainer_config.checkpoint_config.bias,
            use_gradient_checkpointing=self.trainer_config.checkpoint_config.use_gradient_checkpointing,
            random_state=self.trainer_config.checkpoint_config.random_state,
            use_rslora=self.trainer_config.checkpoint_config.use_rsloss,
        )

    def _prepare_dataset(self):
        logger.info("Preparing dataset")

        self.train_formatter = Formatter(
            self.tokenizer,
            training=True
        )

        self.test_formatter = Formatter(
            self.tokenizer,
            training=False
        )

        self.dataset = load_from_disk(
            self.trainer_config.data_config.dataset_path,
        )
        
        self.dataset = self.dataset.map(self.train_formatter, batched=True, num_proc=12)
        

    def _prepare_training(self):
        logger.info("Preparing training")

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['val'],
            dataset_text_field="text",
            max_seq_length=self.trainer_config.checkpoint_config.max_seq_len,
            packing=False,
            dataset_num_proc=24,
            args=TrainingArguments(
                # GPU related arguments
                per_device_train_batch_size=self.trainer_config.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.trainer_config.training_config.per_device_eval_batch_size,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),

                # Optimizer related arguments
                learning_rate=self.trainer_config.training_config.learning_rate,
                weight_decay=self.trainer_config.training_config.weight_decay,
                optim=self.trainer_config.training_config.optim,
                lr_scheduler_type=self.trainer_config.training_config.lr_scheduler_type,

                # Training steps related arguments
                num_train_epochs=self.trainer_config.training_config.num_train_epochs,
                max_steps=self.trainer_config.training_config.max_steps,
                eval_steps=self.trainer_config.training_config.eval_steps,
                logging_steps=self.trainer_config.training_config.logging_steps,
                save_steps=self.trainer_config.training_config.save_steps,
                do_eval=True,
                
                # Other arguments
                output_dir=self.trainer_config.training_config.output_dir,
            )
        )

    def prepare(self):
        self._prepare_dataset()
        self._prepare_training()

    def train(self):

        logger.info("Preparing training")

        self.prepare()

        # Source Unsloth: https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=6bZsfBuZDeCL
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = self.trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        self.model.save_pretrained(
            f'{self.trainer_config.training_config.output_dir}/lora_adapters'
        )

        self.tokenizer.save_pretrained(
            f'{self.trainer_config.training_config.output_dir}/lora_adapters'
        )

        self.model.save_pretrained_merged(
            f'{self.trainer_config.training_config.output_dir}/merged_model',
            save_method='merged_16bit'
        )


