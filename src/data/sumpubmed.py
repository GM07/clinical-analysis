import random
from typing import Dict, List
import nltk
import re
from datasets import load_from_disk, concatenate_datasets

from src.data.synthetic_dataset import SyntheticDataset

PROMPT_TEMPLATE = """
Transform the input sentence by introducing a deliberate inaccuracy. Strategies can include:
- Changing numerical values
- Inverting the meaning
- Using antonyms
- Negating the original statement

Ensure the new sentence remains grammatically correct but semantically different from the original. Only output the transformed sentence, no other text.
Here is the sentence: {sentence}
"""

class SumPubMed(SyntheticDataset):
    """
    These are the pairs of (document, summary) that are used :
    - (text, abstract)
    - (text, shorter_abstract)
    - (abstract, shorter_abstract)    
    """

    KEYWORDS_PATTERN = r'keywords[\s\S]*'

    def __init__(self, path: str):

        self.path = path
        if self.path[-1] != '/':
            self.path += '/'

        self.load()

    def load(self):
        self.raw_data = load_from_disk(self.path)
        self.raw_data = concatenate_datasets([self.raw_data['train'], self.raw_data['test'], self.raw_data['dev']])

    def generate_prompts(self):
        self.positive_data = self.generate_positive_samples()
        self.negative_data = self.generate_negative_samples_prompts()

        # concatenate_datasets will set the value of columns to None if the datasets have different columns
        self.data = concatenate_datasets([self.positive_data, self.negative_data])
        return self.data

    def generate_negative_samples_prompts(self):
        """
        Generate negative samples from the dataset.
        """

        random.seed(42)

        dataset = self.positive_data.map(
            self.duplicate_rows,
            desc="Duplicating rows",
            batched=True,
            batch_size=1,
        )

        dataset = dataset.map(
            self.split_text_for_prompts, 
            desc="Choosing random sentence from summary"
        )

        dataset = dataset.filter(
            lambda example: example["sentence"] is not None, 
            desc="Filtering out examples with not enough sentences"
        )
        dataset = dataset.map(
            lambda example: {'input': PROMPT_TEMPLATE.format(sentence=example["sentence"])},
            desc="Generating chat messages"
        )

        return dataset

    def generate_positive_samples(self):
        dataset1 = self.get_doc_sum_pairs(doc_col="text", sum_col="abstract")
        dataset2 = self.get_doc_sum_pairs(doc_col="text", sum_col="shorter_abstract")
        dataset3 = self.get_doc_sum_pairs(doc_col="abstract", sum_col="shorter_abstract")

        # Concatenate all three datasets
        dataset = concatenate_datasets([dataset1, dataset2, dataset3])

        # Remove keywords from the summaries
        dataset = dataset.map(lambda example: {"summary": self.remove_keywords(example["summary"])})
        return dataset

    def get_doc_sum_pairs(self, doc_col: str, sum_col: str):
        return self.raw_data.map(
            lambda example: {
                "id": example["filename_text"],
                "text": example[doc_col],
                "summary": example[sum_col],
                'factual': [True] * len(example[doc_col])
            },
            remove_columns=self.raw_data.column_names,
            batched=True,
            desc=f"Generating positive samples from {doc_col} and {sum_col}"
        )
    
    def split_text_for_prompts(self, example: dict):
        """
        Split the text into sentences and return a list of sentences.
        """

        if example["factual"] == True:
            # We don't want to split the text for positive samples as we want to keep the original text
            return {
                'sentence': None,
                'before': None,
                'after': None
            }

        sentences = nltk.sent_tokenize(example["summary"])
        if len(sentences) < 2:
            return {
                'sentence': None,
                'before': None,
                'after': None
            }
        
        index = random.choice(range(len(sentences) - 1))

        return {
            'sentence': sentences[index],
            'before': ' '.join(sentences[:index]),
            'after': ' '.join(sentences[index+1:])
        }

    def get_random_sentence(self, text: str, min_length: int = 100):
        """
        Get a random sentence from the text. The nltk sentence tokenizer is used to split the text into sentences.

        Args:
            text (str): The text to get a random sentence from.
            min_length (int): The minimum length of the sentence to return.

        Returns:
            str: A random sentence from the text.
        """
        sentences = nltk.sent_tokenize(text)
        sentences = [sentence for sentence in sentences if len(sentence) > min_length]
        if len(sentences) == 0:
            return None
        return random.choice(sentences)

    def remove_keywords(self, text: str):
        return re.sub(self.KEYWORDS_PATTERN, '', text)

    def duplicate_rows(self, examples: Dict[str, List[str]]):
        return {
            'id': examples['id'] + examples['id'],
            'text': examples['text'] + examples['text'],
            'summary': examples['summary'] + examples['summary'],
            'factual': [True, False]
        }
        
