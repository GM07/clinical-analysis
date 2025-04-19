import logging

from src.training.umed_hal_trainer import UMedHalTrainer
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
    trainer = UMedHalTrainer(trainer_config)
    trainer.train()

if __name__ == '__main__':
    main()
