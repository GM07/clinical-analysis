from datasets import Dataset, load_from_disk, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import logging

logger = logging.getLogger(__name__)

class EvaluatorTrainer:

    DEFAULT_ID2LABEL = {0: 'Nursing/other', 1: 'ECG', 2: 'Radiology', 3: 'Physician '}
    DEFAULT_LABEL2ID = {'Nursing/other': 0, 'ECG': 1, 'Radiology': 2, 'Physician ': 3}

    def __init__(self, model_checkpoint: str, dataset_dict_path: str, label2id: dict = None, id2label: dict = None, local: bool = False):
        """
        Args:
            model_checkpoint: The path to the model checkpoint.
            dataset_dict_path: The path to the dataset dictionary saved on disk by the prepare_dataset method.
            label2id: A dictionary mapping labels to their corresponding IDs. If not provided, the default mapping will be used (DEFAULT_LABEL2ID).
            id2label: A dictionary mapping IDs to their corresponding labels. If not provided, the default mapping will be used (DEFAULT_ID2LABEL).
            local: Whether to load the model locally or from the Huggingface Hub.
        """
        self.model_checkpoint = model_checkpoint
        self.dataset_dict_path = dataset_dict_path
        self.local = local
        if label2id is None:
            self.label2id = self.DEFAULT_LABEL2ID
        else:
            self.label2id = label2id

        if id2label is None:
            self.id2label = self.DEFAULT_ID2LABEL
        else:
            self.id2label = id2label

        self.load_model()
        self.load_dataset()

    def load_dataset(self):
        logger.info(f"Loading dataset from {self.dataset_dict_path}")
        self.dataset_dict = load_from_disk(self.dataset_dict_path)

        def tokenize(samples):
            return self.tokenizer(samples['SUMMARY'], truncation=True, max_length=512)

        self.dataset_dict = self.dataset_dict.map(tokenize, batched=True)

    def load_model(self):
        logger.info(f"Loading model from {self.model_checkpoint}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            local_files_only=self.local
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(
        self, 
        output_dir: str = None, 
        batch_size: int = 16, 
        learning_rate: float = 2e-5, 
        num_epochs: int = 3, 
        weight_decay: float = 0.01, 
        resume_from_checkpoint: bool = None
    ):
        """
        Trains the model on the dataset.

        Args:
            output_dir: The path to save the model.
            batch_size: The batch size.
            learning_rate: The learning rate.
            num_epochs: The number of epochs.
            weight_decay: The weight decay.
            resume_from_checkpoint: If True, resume from checkpoint. Leave None to train from scratch.
        """
        if output_dir is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=1000,
                save_steps=1000
            )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_dict["train"],
            eval_dataset=self.dataset_dict["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.evaluate
        )

        if resume_from_checkpoint is not None:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()

    def evaluate(self, eval_pred):
        predictions, references = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': float(accuracy_score(references, predictions, normalize=True, sample_weight=None))
        }

    
    @staticmethod
    def prepare_dataset(dataset: str | Dataset, output_path: str, label2id: dict):
        """
        Prepares the dataset for training and evaluation by converting it to a Huggingface DatasetDict object, adding labels and splitting it into train and test sets.

        Args:
            dataset : The path to the dataset or the huggingface dataset.
            output_path : The path to save the prepared dataset.
            label2id : A dictionary mapping labels to their corresponding IDs.

        Returns:
            dataset_dict (DatasetDict): A DatasetDict object containing the train and test sets.
        """
        if isinstance(dataset, str):
            dataset = Dataset.from_csv(dataset)

        def get_labels(row):
            return {'label': label2id[row['CATEGORY']]}

        dataset = dataset.map(get_labels, batched=False)

        train_test_valid = dataset.train_test_split(test_size=0.1)
        test_valid = train_test_valid['test'].train_test_split(test_size=0.5)
        final_dataset_dict = DatasetDict(
            {
                'train': train_test_valid['train'].filter(lambda x: x['SUMMARY'] != None),
                'test': test_valid['test'].filter(lambda x: x['SUMMARY'] != None),
                'valid': test_valid['train'].filter(lambda x: x['SUMMARY'] != None)
            }
        )

        final_dataset_dict.save_to_disk(output_path)

        return final_dataset_dict

