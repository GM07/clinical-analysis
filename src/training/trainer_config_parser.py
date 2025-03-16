import sys
import yaml

from src.training.trainer_config import CheckpointConfig, DataConfig, TrainingConfig, TrainerConfig

import logging  

logger = logging.getLogger(__name__)

class TrainerConfigParser:

    def parse(self, config_path: str = None) -> TrainerConfig:
        if config_path is None:
            logger.info("No config path provided, using first argument as config path")
            config_path = sys.argv[1]
        

        with open(config_path, 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file {config_path}: {e}")
        
        default_config = TrainerConfig()
        
        config_dict = {
            'checkpoint_config': vars(default_config.checkpoint_config),
            'data_config': vars(default_config.data_config),
            'training_config': vars(default_config.training_config)
        }
        
        # Update with values from yaml file
        for section in yaml_config:
            if section in config_dict and yaml_config[section]:
                valid_keys = set(config_dict[section].keys())
                for key, value in yaml_config[section].items():
                    if key in valid_keys:
                        config_dict[section][key] = value
                    else:
                        warning = f"Unknown parameter '{key}' in section '{section}' will be ignored"
                        logger.warning(warning)
            else:
                logger.warning(f"Unknown section '{section}' in config file {config_path}")

        c = TrainerConfig(
            checkpoint_config=CheckpointConfig(**config_dict['checkpoint_config']),
            data_config=DataConfig(**config_dict['data_config']),
            training_config=TrainingConfig(**config_dict['training_config'])
        )
        # Validate required fields
        for config_class_name, config_class in c.__dict__.items():
            # Get the class type to access field metadata
            class_type = type(config_class)
            
            # Iterate through fields in the class
            for field_name, field_value in config_class.__dict__.items():
                # Check if this field has metadata and is required
                if field_name in class_type.__dataclass_fields__:
                    field_metadata = class_type.__dataclass_fields__[field_name].metadata
                    if field_metadata.get('required', False) and field_value is None:
                        logger.error(f"Missing required field '{field_name}' in section '{config_class_name}' in config file {config_path}")
                        raise ValueError(f"Missing required field '{field_name}' in section '{config_class_name}' in config file {config_path}")

        return c
