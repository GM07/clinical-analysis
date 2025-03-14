from dataclasses import dataclass
from typing import Dict, Union

from transformers import BitsAndBytesConfig

@dataclass
class LoadingConfig:
    pad_equals_eos: bool = True
    use_quantization: bool = False
    quantization_config: BitsAndBytesConfig = None
    bf16: bool = True
    device_map: Union[str, Dict[int, str]] = 'auto'
    padding_side: str = 'left'
    local_files_only: bool = True
