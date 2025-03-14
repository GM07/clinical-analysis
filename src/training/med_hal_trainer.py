from datasets import load_from_disk
from src.models.loading_config import LoadingConfig
from src.models.utils import load_model, load_tokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

import logging

logger = logging.getLogger(__name__)

from src.training.formatter import Formatter

class MedHALTrainer:
    def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.dataset_path = dataset_path

    def load(self, loading_config: LoadingConfig):
        self.loading_config = loading_config
        self.model = load_model(self.model_checkpoint, loading_config)
        self.tokenizer = load_tokenizer(self.tokenizer_checkpoint, loading_config)

        logger.info(f'Loading dataset from {self.dataset_path}')
        self.dataset = load_from_disk(self.dataset_path)

    def train(self, output_dir: str, batch_size: int = 1, resume_from_checkpoint: str = None):

        config = LoraConfig(
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            task_type='CAUSAL_LM',
            use_rslora=True,
            bias = 'none',
        )

        # self.model = prepare_model_for_kbit_training(self.model)

        # self.model = get_peft_model(self.model, config)
        # logger.info(f'Model has {self.model.num_parameters()} trainable parameters')


        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            # max_steps=100,
            # lr_scheduler_type='cosine',
            # warmup_steps=5,
            # optim="adamw_8bit",
            # logging_dir=f'{output_dir}/logs',
            # logging_strategy='steps',
            # logging_steps=10,
            # save_strategy='steps',
            # save_steps=10,
            # eval_strategy='steps',
            # eval_steps=10,
            # bf16=True,
            # do_eval=True,
            # seed=42,
            # packing=True,
            max_seq_length=8192,
        )

        logger.info('Creating trainer')
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset['train'].select(range(2)),
            eval_dataset=self.dataset['train'].select(range(2)),
            tokenizer=self.tokenizer,
            formatting_func=Formatter(),
            args=args,
            peft_config=config,
        )

        logger.info('Training model')
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info('Saving model')
        trainer.save_model(f'{output_dir}/peft_model_only')

        # base_model = load_model(self.model_checkpoint, self.loading_config)
        # peft_model = PeftModel.from_pretrained(base_model, f'{output_dir}/peft_model_only')
        # merged_model = peft_model.merge_and_unload()
        # merged_model.save_pretrained(f'{output_dir}/model')

    def evaluate(self):
        pass


# class ImprovedMedHALTrainer:
#     def __init__(self, model_checkpoint, tokenizer_checkpoint, dataset_path, max_seq_length: int = 8192):
#         self.model_checkpoint = model_checkpoint
#         self.tokenizer_checkpoint = tokenizer_checkpoint
#         self.dataset_path = dataset_path
#         self.max_seq_length = max_seq_length
        
#     def load(self):

#         logger.info(f'Loading model from {self.model_checkpoint}')
#         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
#             self.model_checkpoint,
#             max_seq_length=self.max_seq_length,
#             dtype=None,
#             load_in_4bit=True,
#             local_files_only=True,
#         )

#         logger.info('Configuring LoRA')
#         self.model = FastLanguageModel.get_peft_model(
#             self.model,
#             r = 16, # Suggested 8, 16, 32, 64, 128
#             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                             "gate_proj", "up_proj", "down_proj",],
#             lora_alpha = 16,
#             lora_dropout = 0, # Supports any, but = 0 is optimized
#             bias = "none",    # Supports any, but = "none" is optimized
#             # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#             use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#             random_state = 42,
#             use_rslora = False,  # We support rank stabilized LoRA
#             loftq_config = None, # And LoftQ
#         )

#         logger.info(f'Loading dataset from {self.dataset_path}')
#         self.dataset = load_from_disk(self.dataset_path)

#     def train(self, output_dir: str, batch_size: int = 1):
#         formatter = Formatter()

#         def format(example):
#             return {'text':formatter(example)}
        
#         self.formatted_dataset = self.dataset.select(range(10)).map(format, batched=True)


#         trainer = SFTTrainer(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             train_dataset=self.formatted_dataset,
#             dataset_text_field="text",
#             max_seq_length=self.max_seq_length,
#             dataset_num_proc=2,
#             packing=False, # Can make training 5x faster for short sequences.
#             args=TrainingArguments(
#                 per_device_train_batch_size=2,
#                 gradient_accumulation_steps=4,
#                 warmup_steps=5,
#                 num_train_epochs=1, # Set this for 1 full training run.
#                 # max_steps=60,
#                 learning_rate=2e-4,
#                 fp16=not is_bfloat16_supported(),
#                 bf16=is_bfloat16_supported(),
#                 logging_steps=10,
#                 optim="adamw_8bit",
#                 weight_decay=0.01,
#                 lr_scheduler_type="linear",
#                 seed=42,
#                 output_dir=output_dir,
#                 report_to="none", # Use this for WandB etc
#             ),
#         )

#         stats = trainer.train()

#         self.model.save_pretrained_merged(f'{output_dir}/model', self.tokenizer, save_method='merged_16bit')
