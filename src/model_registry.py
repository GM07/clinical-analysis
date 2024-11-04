from dataclasses import dataclass
import os
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

logger = logging.getLogger(__name__)

@dataclass
class LoadingConfig:
    pad_equals_eos: bool = True
    use_quantization: bool = False
    quantization_config: BitsAndBytesConfig = None
    bf16: bool = True
    device_map = 'auto'
    padding_side = 'left'

@dataclass
class ModelNames:
    pass

class ModelRegistry:
    """
    Helper class to load HuggingFace models
    """

    def __init__(self, local: bool, local_folder_path: str = ''):
        self.local = local
        self.local_folder_path = local_folder_path
        self.model_names = ModelNames()

        self.get_models()

    def get_models(self):
        """
        Will scan the `local_folder_path` for all folders and retrieve the name
        of those folders. These paths will be added as attributes to the `model_names`
        attribute
        """

        if self.local_folder_path is None or len(self.local_folder_path) == 0:
            return
        
        assert os.path.exists(self.local_folder_path), f"The local path provided for the model registry is not valid : {self.local_folder_path}"

        model_names = [f.name for f in os.scandir(self.local_folder_path) if f.is_dir()]
        logger.info(f'Found models : {model_names}')
        for model_name in model_names:
            setattr(self.model_names, model_name.replace('-', '_'), model_name)

    def load_checkpoint(self, checkpoint: str, loading_config: LoadingConfig = LoadingConfig()):
        """
        Loads a checkpoint (model and tokenizer) using a model name

        Args:
            checkpoint: Name of the model as shown in the HuggingFace website
            loading_config: Config on how to load the model

        Returns
        Tuple containing the model and the tokenizer
        """
        return self.load_model(checkpoint, loading_config), self.load_tokenizer(checkpoint, loading_config)

    def load_tokenizer(self, checkpoint, loading_config: LoadingConfig = LoadingConfig()):
        """
        Loads a tokenizer using a model name from the registry

        Args:
            checkpoint: Name of the tokenizer as shown in the HuggingFace website
            loading_config: Config on how to load the tokenizer

        Returns
        Tokenizer object 
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.get_full_checkpoint(checkpoint), 
            local_files_only=self.local,
            padding_side=loading_config.padding_side
        )

        if loading_config.pad_equals_eos:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(self, checkpoint, loading_config: LoadingConfig = LoadingConfig()):
        """
        Loads a model using a model name from the registry

        Args:
            checkpoint: Name of the model as shown in the HuggingFace website
            loading_config: Config on how to load the model

        Returns
        Model object 
        """
        config = None
        if loading_config.use_quantization:
            if loading_config.quantization_config:
                config = loading_config.quantization_config
            else:
                compute_dtype = getattr(torch, "float16")
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                )

        return AutoModelForCausalLM.from_pretrained(
            self.get_full_checkpoint(checkpoint), 
            local_files_only=self.local, 
            trust_remote_code=True,
            quantization_config=config,
            device_map=loading_config.device_map,
            torch_dtype=torch.bfloat16 if loading_config.bf16 else torch.float16,
        )

    def get_full_checkpoint(self, checkpoint: str):
        """Will append the path of the registry in the case of a local registry"""
        if self.local:
            return self.local_folder_path + checkpoint
        return checkpoint
