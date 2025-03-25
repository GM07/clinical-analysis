import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

from src.models.loading_config import LoadingConfig

logger = logging.getLogger(__name__)

def get_4bit_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_storage=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=False,
    )

def get_8bit_quantization_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_8bit_quant_type='nf4',
        # bnb_8bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # bnb_8bit_quant_storage=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # bnb_8bit_use_double_quant=False,
    )

def load_model(checkpoint, loading_config: LoadingConfig = LoadingConfig()):
    """
    Loads a model using a model name from the registry

    Args:
        checkpoint: Name of the model as shown in the HuggingFace website
        loading_config: Config on how to load the model

    Returns
    Model object 
    """
    logger.info(f'Loading model from {checkpoint} with config {loading_config}')
    config = None
    if loading_config.use_quantization:
        if loading_config.quantization_config:
            config = loading_config.quantization_config
        else:
            config = get_4bit_quantization_config()
    logger.info(f"Config: {config}")
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=loading_config.local_files_only, 
        trust_remote_code=True,
        quantization_config=config,
        device_map=loading_config.device_map,
        torch_dtype=torch.bfloat16 if loading_config.bf16 else torch.float16,
    )

def load_tokenizer(checkpoint, loading_config: LoadingConfig = LoadingConfig()):
    """
    Loads a tokenizer using a model name from the registry

    Args:
        checkpoint: Name of the tokenizer as shown in the HuggingFace website
        loading_config: Config on how to load the tokenizer

    Returns
    Tokenizer object 
    """
    logger.info(f'Loading tokenizer from {checkpoint} with config {loading_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        local_files_only=loading_config.local_files_only,
        padding_side=loading_config.padding_side
    )

    if loading_config.pad_equals_eos:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model_empty(checkpoint, model_name, model_class, loading_config: LoadingConfig = LoadingConfig()):
    """
    Loads a model using a model name from the registry

    Args:
        checkpoint: Name of the model as shown in the HuggingFace website
        loading_config: Config on how to load the model

    Returns
    Model object 
    """
    logger.info(f'Loading model from {checkpoint} with config {loading_config}')
    config = None
    if loading_config.use_quantization:
        if loading_config.quantization_config:
            config = loading_config.quantization_config
        else:
            config = get_4bit_quantization_config()
    logger.info(f"Config: {config}")

    # 2. Load the model configuration:
    model_config = AutoConfig.from_pretrained(model_name)

    # 3. Instantiate the model with empty weights:
    with init_empty_weights():
        model = model_class(model_config)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint,
        # local_files_only=loading_config.local_files_only, 
        # trust_remote_code=True,
        # quantization_config=config,
        # device_map=loading_config.device_map,
        dtype=torch.bfloat16 if loading_config.bf16 else torch.float16,
    )

    return model
