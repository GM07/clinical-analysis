import os
import logging
import datetime

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from src.data.formatter import Formatter
from src.training.trainer_config import TrainerConfig


logger = logging.getLogger(__name__)

class UMedHalTrainer:

    RESPONSE_TEMPLATE_CONTEXT = "### Factual"

    def __init__(self, trainer_config: TrainerConfig):
        self.trainer_config = trainer_config
        self.load_checkpoint()

    def load_checkpoint(self):
        logger.info(f"Loading checkpoint : {self.trainer_config.checkpoint_config.model_checkpoint}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.trainer_config.checkpoint_config.model_checkpoint,
            max_seq_length=self.trainer_config.checkpoint_config.max_seq_len,
            dtype=None,
            load_in_4bit=self.trainer_config.checkpoint_config.load_in_4bit,
        )

        logger.info(f'Model loaded : {self.model}')

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
        
    def _prepare_collator(self):
        logger.info("Preparing collator")

        encoded_response_template = self.tokenizer.encode(
            self.RESPONSE_TEMPLATE_CONTEXT, add_special_tokens=False
        )

        # Evaluate if the response template is present in the first and last examples
        first_example = self.dataset['train'][-1]["text"]
        logger.info(f"Example formatted: {first_example}")
        first_example_ids = self.tokenizer.encode(first_example, add_special_tokens=False)

        assert self.RESPONSE_TEMPLATE_CONTEXT in first_example
        assert encoded_response_template[0] in first_example_ids
        assert encoded_response_template[-1] in first_example_ids

        last_example = self.dataset['train'][-1]["text"]
        last_example_ids = self.tokenizer.encode(last_example, add_special_tokens=False)

        assert self.RESPONSE_TEMPLATE_CONTEXT in last_example
        assert encoded_response_template[0] in last_example_ids
        assert encoded_response_template[-1] in last_example_ids

        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=encoded_response_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def _prepare_training(self):
        logger.info("Preparing training")

        training_folder = self.trainer_config.training_config.output_dir
        if training_folder[-1] != '/':
            training_folder += '/'
        training_folder += '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        os.makedirs(training_folder, exist_ok=True)

        logger.info(f'Created folder for training at {training_folder}')

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['val'],
            data_collator=self.data_collator,
            # data_collator=self.data_collator,
            # peft_config=peft_config,
            # packing=False,
            args=SFTConfig(
                # GPU related arguments
                per_device_train_batch_size=self.trainer_config.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.trainer_config.training_config.per_device_eval_batch_size,
                gradient_checkpointing=self.trainer_config.training_config.use_gradient_checkpointing,
                gradient_accumulation_steps=self.trainer_config.training_config.gradient_accumulation_steps,
                bf16=True,

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
                output_dir=training_folder,
                max_seq_length=self.trainer_config.checkpoint_config.max_seq_len,
                dataset_num_proc=12,
            )
        )

    def prepare(self):
        self._prepare_dataset()
        self._prepare_collator()
        self._prepare_training()

    def train(self):

        self.prepare()
        logger.info("Training")
        log_trainable_parameters(self.model)
        
        stats = self.trainer.train()

        print(stats)

        self.trainer.save_model(
            f'{self.trainer_config.training_config.output_dir}/lora_adapters'
        )

        self.tokenizer.save_pretrained(
            f'{self.trainer_config.training_config.output_dir}/lora_adapters'
        )

def log_trainable_parameters(model) -> float:
    """
    Logs the percentage of trainable parameters in a Hugging Face model.

    Args:
        model: The Hugging Face PreTrainedModel instance (after LoRA or other
               parameter-efficient fine-tuning).

    """
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(f"Total parameters in the model (after LoRA): {total_params:,}")
    logger.info(f"Trainable parameters (LoRA): {trainable_params:,}")
    logger.info(f"Percentage of trainable parameters: {(trainable_params / total_params) * 100:.2f}%")
