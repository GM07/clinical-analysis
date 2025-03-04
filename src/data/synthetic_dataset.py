

from abc import abstractmethod

class SyntheticDataset:

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate_prompts(self):
        pass
