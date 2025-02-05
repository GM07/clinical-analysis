
from datasets import Dataset

class HalDataset:

    COLUMNS = ['internal_id', 'context', 'statement', 'factual', 'dataset']

    def __init__(self, path: str):
        self.path = path

        self.load()

    def load(self):
        self.data = Dataset.from_csv(self.path)
        return self.data

    def verify_columns(self):
        for column in self.COLUMNS:
            if column not in self.data.column_names:
                raise ValueError(f"Column {column} not found in dataset")
