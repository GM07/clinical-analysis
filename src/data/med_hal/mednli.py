import os

import pandas as pd

class MedNLI:

    def __init__(self, path: str, train_file_relative_path: str = 'mli_train_v1.jsonl'):
        self.path = path
        self.train_file_relative_path = train_file_relative_path
        self.train_file_path = os.path.join(path, train_file_relative_path)

        self.load()

    def load(self):
        self.raw_data = pd.read_json(self.train_file_path, lines=True)
        return self.raw_data

    def generate_prompts(self):
        # Remove all columns except, sentence1, sentence2, gold_label and pairID
        self.data = self.raw_data[['sentence1', 'sentence2', 'gold_label', 'pairID']]

        # Remove all rows where gold_label is not 'entailment' or 'contradiction'
        self.data = self.data[self.data['gold_label'].isin(['entailment', 'contradiction'])]
        self.data['factual'] = self.data['gold_label'].apply(lambda x: True if x == 'entailment' else False)

        # Statement 1 is the premise (context)
        # Statement 2 is the hypothesis (statement)
        self.data = self.data.rename(columns={'sentence1': 'context', 'sentence2': 'statement'})

        self.data = self.data.drop(columns=['gold_label'])
        self.data = self.data.rename(columns={'pairID': 'internal_id'})

        self.data['dataset'] = 'mednli'

        return self.data


    def save(self, path: str):
        self.data.to_csv(path, index=False)
