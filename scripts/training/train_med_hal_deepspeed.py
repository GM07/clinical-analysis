from src.training.med_hal_trainer_deepspeed import MedHalTrainerDeepSpeed
from src.training.trainer_config_parser import TrainerConfigParser

from accelerate import Accelerator
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[PROGRAM] [%(levelname)s] - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

accelerator = Accelerator()

def main():
    trainer_config = TrainerConfigParser().parse()
    print(trainer_config)
    trainer = MedHalTrainerDeepSpeed(trainer_config, accelerator=accelerator)
    trainer.train()

    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
