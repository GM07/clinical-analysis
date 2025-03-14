import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from src.models.loading_config import LoadingConfig

logger = logging.getLogger(__name__)

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
            compute_dtype = getattr(torch, "bfloat16")
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )

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
