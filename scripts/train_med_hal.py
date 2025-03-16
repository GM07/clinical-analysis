import logging

from src.training.med_hal_trainer import MedHalTrainer
from src.training.trainer_config import TrainerConfig
from src.training.trainer_config_parser import TrainerConfigParser

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)


def main():
    trainer_config = TrainerConfigParser().parse()
    print(trainer_config)
    trainer = MedHalTrainer(trainer_config)
    trainer.train()

if __name__ == '__main__':
    main()
