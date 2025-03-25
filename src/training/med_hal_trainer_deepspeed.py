import os
import logging
import datetime

import torch
import torch.distributed as dist
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig, LoftQConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import deepspeed

from src.data.formatter import Formatter
from src.models.loading_config import LoadingConfig
from src.models.utils import get_4bit_quantization_config, get_8bit_quantization_config, load_model, load_tokenizer
from src.training.trainer_config import TrainerConfig


logger = logging.getLogger(__name__)

class MedHalTrainerDeepSpeed:

    RESPONSE_TEMPLATE_CONTEXT = "### Factual"

    def __init__(self, trainer_config: TrainerConfig, accelerator: Accelerator):
        self.trainer_config = trainer_config
        self.accelerator = accelerator
        set_seed(42)  # Add seed for reproducibility
        self.load_checkpoint()

    def load_checkpoint(self):
        logger.info(f"Loading checkpoint : {self.trainer_config.checkpoint_config.model_checkpoint}")

        # Simplified quantization and loading
        use_quantization = self.trainer_config.checkpoint_config.load_in_4bit or self.trainer_config.checkpoint_config.load_in_8bit
        if self.trainer_config.training_config.loftq_bits:
            use_quantization = False

        quantization_config = None
        if self.trainer_config.checkpoint_config.load_in_4bit:
            quantization_config = get_4bit_quantization_config()
        elif self.trainer_config.checkpoint_config.load_in_8bit:
            quantization_config = get_8bit_quantization_config()
            
        loading_config = LoadingConfig(
            use_quantization=use_quantization,
            quantization_config=quantization_config,
            pad_equals_eos=False,
            padding_side='right',
        )

        self.model: AutoModelForCausalLM = load_model(
            self.trainer_config.checkpoint_config.model_checkpoint,
            loading_config=loading_config,
        )

        self.model.max_seq_length = self.trainer_config.checkpoint_config.max_seq_len

        self.tokenizer: AutoTokenizer = load_tokenizer(
            self.trainer_config.checkpoint_config.model_checkpoint,
            loading_config=loading_config,
        )

        if self.tokenizer.pad_token is None:
            logger.warning("No pad token found, setting it to <finetune-pad-token>")
            self.tokenizer.add_special_tokens({'pad_token': '<finetune-pad-token>'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_distributed_environment(self):
        """
        Explicitly initialize the distributed environment and device mapping.
        This helps resolve NCCL process group initialization issues.
        """
        # Get local rank and world size
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Set the device explicitly
        torch.cuda.set_device(local_rank)

        # Initialize the process group with explicit device ID
        dist.init_process_group(
            backend='nccl', 
            init_method='env://', 
            world_size=world_size, 
            rank=local_rank
        )

        logger.info(f"Initialized process group: Local Rank {local_rank}, World Size {world_size}")


    def _prepare_dataset(self):
        logger.info("Preparing dataset")

        self.train_formatter = Formatter(self.tokenizer, training=True)
        self.test_formatter = Formatter(self.tokenizer, training=False)

        self.dataset = load_from_disk(self.trainer_config.data_config.dataset_path)
        self.dataset = self.dataset.map(self.train_formatter, batched=True, num_proc=12)
        
    def _prepare_collator(self):
        logger.info("Preparing collator")

        encoded_response_template = self.tokenizer.encode(
            self.RESPONSE_TEMPLATE_CONTEXT, add_special_tokens=False
        )

        # Validate response template
        first_example = self.dataset['train'][0]["text"]
        last_example = self.dataset['train'][-1]["text"]
        
        assert self.RESPONSE_TEMPLATE_CONTEXT in first_example
        assert self.RESPONSE_TEMPLATE_CONTEXT in last_example

        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=encoded_response_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def _prepare_peft_model(self):
        logger.info("Preparing PEFT model")

        # Prepare LoRA configuration
        use_loftq = self.trainer_config.training_config.loftq_bits is not None
        lora_config_args = {
            'r': self.trainer_config.training_config.r,
            'lora_alpha': self.trainer_config.training_config.lora_alpha,
            'lora_dropout': self.trainer_config.training_config.lora_dropout,
            'bias': self.trainer_config.training_config.bias,
            'use_rslora': self.trainer_config.training_config.use_rslora,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }

        if use_loftq:
            lora_config_args['loftq_config'] = LoftQConfig(loftq_bits=self.trainer_config.training_config.loftq_bits)
            lora_config_args['init_lora_weights'] = 'loftq'

        peft_config = LoraConfig(**lora_config_args)
        
        # Ensure model is on the correct device and not in meta device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # Handling potential meta device scenario
        if any(p.device == torch.device('meta') for p in self.model.parameters()):
            logger.warning("Model contains meta tensors. Attempting to convert.")
            try:
                # Use to_empty to handle meta tensors
                self.model = self.model.to_empty(device=device)
            except Exception as e:
                logger.error(f"Failed to convert meta tensors: {e}")
                raise

        self.model = get_peft_model(self.model, peft_config)

        # self.model = get_peft_model(self.model, peft_config)
        
        # Prepare model for DeepSpeed (important for ZeRO-3)
        if self.accelerator.distributed_type == 'DEEPSPEED':
            self.model = self._prepare_deepspeed_model(self.model)

    def _prepare_deepspeed_model(self, model):
        logger.info("Preparing model for DeepSpeed")
        
        # Required for DeepSpeed ZeRO-3
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        return model

    def _prepare_training(self):
        logger.info("Preparing training")

        # Create training output directory
        training_folder = self.trainer_config.training_config.output_dir
        training_folder = os.path.join(
            training_folder, 
            datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        )
        os.makedirs(training_folder, exist_ok=True)

        logger.info(f'Created folder for training at {training_folder}')

        # Prepare PEFT and model for training
        self._prepare_peft_model()

        # Ensure model is on the correct device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.model = self.model.to(torch.device(f"cuda:{local_rank}"))

        # Prepare DeepSpeed configuration
        deepspeed_config = {
            "train_batch_size": self.trainer_config.training_config.per_device_train_batch_size * dist.get_world_size(),
            "steps_per_print": 1000,
            "optimizer": {
                "type": self.trainer_config.training_config.optim,
                "params": {
                    "lr": self.trainer_config.training_config.learning_rate,
                    "weight_decay": self.trainer_config.training_config.weight_decay,
                }
            },
            "scheduler": {
                "type": self.trainer_config.training_config.lr_scheduler_type,
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            "bf16": {"enabled": True},
        }

        # Prepare model with DeepSpeed
        try:
            self.model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=deepspeed_config,
                model_parameters=self.model.parameters(),
            )
        except Exception as e:
            logger.error(f"DeepSpeed initialization failed: {e}")
            raise

        # Create trainer with DeepSpeed
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['val'],
            data_collator=self.data_collator,
            args=SFTConfig(
                # GPU and DeepSpeed configurations
                per_device_train_batch_size=self.trainer_config.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.trainer_config.training_config.per_device_eval_batch_size,
                gradient_accumulation_steps=8,
                bf16=True,

                # Optimizer settings
                learning_rate=self.trainer_config.training_config.learning_rate,
                weight_decay=self.trainer_config.training_config.weight_decay,
                optim=self.trainer_config.training_config.optim,
                lr_scheduler_type=self.trainer_config.training_config.lr_scheduler_type,

                # Training steps
                num_train_epochs=self.trainer_config.training_config.num_train_epochs,
                max_steps=self.trainer_config.training_config.max_steps,
                eval_steps=self.trainer_config.training_config.eval_steps,
                logging_steps=self.trainer_config.training_config.logging_steps,
                save_steps=self.trainer_config.training_config.save_steps,
                do_eval=True,
                
                # Output and sequence configuration
                output_dir=training_folder,
                max_seq_length=self.trainer_config.checkpoint_config.max_seq_len,
            )
        )

    def prepare(self):
        # Ensure all processes go through preparation
        self._prepare_dataset()
        self._prepare_collator()
        self._prepare_training()


    def train(self):
        try:
            # Prepare training for all processes
            self.prepare()
            
            logger.info("Starting training")
            stats = self.trainer.train()
            print(stats)

            # Save model and tokenizer
            if dist.get_rank() == 0:
                save_path = os.path.join(
                    self.trainer_config.training_config.output_dir, 
                    'lora_adapters'
                )
                self.trainer.save_model(save_path)
                self.tokenizer.save_pretrained(save_path)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Properly clean up the distributed environment
            dist.destroy_process_group()
