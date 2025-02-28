import ast
from datasets import Dataset as HuggingFaceDataset
import h5py

class ChatInferenceDataset:
    """
    Class to load and prepare a dataset for chat inference. The problem with using csv files for inference
    is that if the chat conversation is saved in a column in order to be sent to the model, the chat will be
    saved as a string and not as a list of dictionaries.

    This class will load a dataset and save a new dataset with the chat conversation as a list of dictionaries.
    """

    def __init__(self, csv_dataset_path: str):
        self.csv_dataset_path = csv_dataset_path
        self.load()

    def load(self):
        self.dataset = HuggingFaceDataset.from_csv(self.csv_dataset_path)

    def save(self, output_path: str, chat_column: str = 'prompt'):
        with h5py.File(output_path, 'w') as h5f:
            chats = list(map(lambda x: ast.literal_eval(x), self.dataset[chat_column]))
            h5f[chat_column] = chats
