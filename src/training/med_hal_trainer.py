from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.training.formatter import Formatter

class MedHALTrainer:
    def __init__(self, model_checkpoint, tokenizer_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint

        self.load()

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

    def train(self, dataset_path: str, output_dir: str, batch_size: int = 4):

        dataset = load_dataset(dataset_path)

        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            optim="adamw",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            save_strategy="steps",
            save_steps=100,
            eval_strategy='steps',
            eval_steps=10,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            tokenizer=self.tokenizer,
            formatting_func=Formatter(),
            args=args
        )

        trainer.train()
        trainer.save_model(f'{output_dir}/model')

    def evaluate(self):
        pass
