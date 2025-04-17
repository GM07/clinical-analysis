from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM

from src.models.utils import load_model, load_tokenizer

class ModelMerger:
    """
    Merges the adapters from a training checkpoint with the base model
    """

    def __init__(self, base_path: str, checkpoint_path: str) -> None:
        self.base_path = base_path
        self.checkpoint_path = checkpoint_path

        self.load()

    def load(self):
        self.model = load_model(self.base_path)
        self.tokenizer = load_tokenizer(self.base_path)

        self.peft_model = PeftModelForCausalLM.from_pretrained(self.model, self.checkpoint_path)

    def merge(self, save_path: str):
        self.peft_model = self.peft_model.merge_and_unload()
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
