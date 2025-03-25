from typing import Optional
import yaml
import sys
import warnings

from dataclasses import dataclass, field

@dataclass
class CheckpointConfig:
    model_checkpoint: str = field(
        default=None,
        metadata={
            'help': 'The model checkpoint to use (required).',
            'required': True,
            'is_path': True # Will allow dynamic expansion of the path (variables like $HOME can be used)
        }
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            'help': 'Whether to load the model in 4-bit (defaults to False).'
        }
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={
            'help': 'Whether to load the model in 8-bit (defaults to False).'
        }
    )

    max_seq_len: int = field(
        default=8192,
        metadata={
            'help': 'The maximum sequence length to use (defaults to 8192).'
        }
    )
    dtype: str = field(
        default=None,
        metadata={
            'help': 'The dtype to use when loading the model (None for auto-detection, defaults to None).'
        }
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            'help': 'Whether to trust remote code (defaults to True).'
        }
    )

    
@dataclass
class DataConfig:
    dataset_path: str = field(
        default=None,
        metadata={
            'help': 'The path to the dataset (required).',
            'required': True,
            'is_path': True
        }
    )
    train_split: str = field(
        default='train',
        metadata={
            'help': 'The train split to use (defaults to "train").'
        }
    )
    val_split: str = field(
        default='val',
        metadata={
            'help': 'The validation split to use (defaults to "val").'
        }
    )
    test_split: str = field(
        default='test',
        metadata={
            'help': 'The test split to use (defaults to "test").'
        }
    )

@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            'help': 'The per-device training batch size (defaults to 8).'
        }
    )

    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            'help': 'The per-device evaluation batch size (defaults to 8).'
        }
    )

    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            'help': 'The number of gradient accumulation steps (defaults to 4).'
        }
    )

    logging_steps: int = field(
        default=10,
        metadata={
            'help': 'The number of logging steps (defaults to 10).'
        }
    )

    num_train_epochs: int = field(
        default=1,
        metadata={
            'help': 'The number of training epochs (defaults to 1).'
        }
    )

    max_steps: int = field(
        default=-1,
        metadata={
            'help': 'The maximum number of steps to use (defaults to None). Will override num_train_epochs if set.'
        }
    )

    save_steps: int = field(
        default=100,
        metadata={
            'help': 'The number of steps to save the model (defaults to 100).'
        }
    )

    eval_steps: int = field(
        default=100,
        metadata={
            'help': 'The number of steps to evaluate the model (defaults to 10).'
        }
    )

    learning_rate: float = field(
        default=2e-4,
        metadata={
            'help': 'The learning rate to use (defaults to 2e-4).'
        }
    )
    optim: str = field(
        default="adamw",
        metadata={
            'help': 'The optimizer to use (defaults to "adamw").'
        }
    )

    weight_decay: float = field(
        default=0.01,
        metadata={
            'help': 'The weight decay to use (defaults to 0.01).'
        }
    )
    
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            'help': 'The learning rate scheduler type to use (defaults to "linear").'
        }
    )
    
    seed: int = field(
        default=42,
        metadata={
            'help': 'The seed to use (defaults to 42).'
        }
    )
    
    output_dir: str = field(
        default="outputs",
        metadata={
            'help': 'The output directory to use (defaults to "outputs").'
        }
    )

    use_lora: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use LoRA (defaults to False).'
        }
    )

    r: int = field(
        default=64,
        metadata={
            'help': 'The rank of the LoRA layers (defaults to 16).'
        }
    )
    lora_alpha: float = field(
        default=16,
        metadata={
            'help': 'The alpha parameter for the LoRA layers (defaults to 16).'
        }
    )

    lora_dropout: float = field(
        default=0.0,
        metadata={
            'help': 'The dropout rate for the LoRA layers (defaults to 0.0).'
        }
    )
    bias: str = field(
        default="none",
        metadata={
            'help': 'The bias to use for the LoRA layers (defaults to "none").'
        }
    )
    use_gradient_checkpointing: str | bool = field(
        default=True,
        metadata={
            'help': 'Whether to use gradient checkpointing (defaults to "True").'
        }
    )
    
    random_state: int = field(
        default=42,
        metadata={
            'help': 'The random state to use (defaults to 42).'
        }
    )

    use_rslora: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use Rank-Stablized LoRA (defaults to False).'
        }
    )

    loftq_bits: int | None = field(
        default=None,
        metadata={
            'help': 'The number of bits to use for the LoFTQ (defaults to None).'
        }
    )
    

@dataclass
class TrainerConfig:
    checkpoint_config: CheckpointConfig = field(
        default_factory=CheckpointConfig
    )
    data_config: DataConfig = field(
        default_factory=DataConfig
    )
    training_config: TrainingConfig = field(
        default_factory=TrainingConfig
    )
