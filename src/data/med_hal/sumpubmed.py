from collections import defaultdict
import random
from typing import Dict, List
import nltk
import re

from datasets import load_from_disk, concatenate_datasets, Dataset

from src.data.synthetic_dataset import SyntheticDataset

PROMPT_TEMPLATE = """You will be given a text and a sentence that was extracted from the text.
Your task is to transform the sentence by introducing a deliberate inaccuracy. Strategies can include:
- Changing numerical values
- Inverting the meaning of the sentence
- Using antonyms
- Negating the original statement

Text: {text}
Sentence: {sentence}

Ensure the new sentence remains grammatically correct but semantically different from the original. The new sentence must contradict the original sentence. Only output the transformed sentence without any additional text.
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

    def generate_prompts(self, output_path: str = None):
        self.positive_data: Dataset = self.generate_positive_samples()
        self.negative_data = self.generate_negative_samples_prompts()

        # concatenate_datasets will set the value of columns to None if the datasets have different columns
        self.data = concatenate_datasets([self.positive_data, self.negative_data])

        if output_path is not None:
            positive_path = output_path.replace('.csv', '_positive.csv')
            negative_path = output_path.replace('.csv', '_negative.csv')
            self.positive_data.to_csv(positive_path, index=False)
            self.negative_data.to_csv(negative_path, index=False)

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
        )

        dataset = dataset.map(
            self.split_text_for_prompts, 
            desc="Choosing random sentence from summary",
            batched=True,
        )

        dataset = dataset.filter(
            lambda example: len(example["sentence"]) > 0, 
            desc="Filtering out examples with not enough sentences"
        )

        dataset = dataset.map(
            lambda example: {'input': [PROMPT_TEMPLATE.format(text=text, sentence=sentence) for text, sentence in zip(example['text'], example['sentence'])]},
            desc="Generating chat messages",
            batched=True
        )

        # Positive and negative data must have the same columns as they will be concatenated
        # We put an empty string so that when reading the file, the column's type is string
        self.positive_data = self.positive_data.add_column('input', [''] * len(self.positive_data))
        self.positive_data = self.positive_data.add_column('sentence', [''] * len(self.positive_data))
        self.positive_data = self.positive_data.add_column('before', [''] * len(self.positive_data))
        self.positive_data = self.positive_data.add_column('after', [''] * len(self.positive_data))

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

        final_dict = defaultdict(list)

        for i in range(len(example['factual'])):

            if example['factual'][i] == True:
                # We don't want to split the text for positive samples as we want to keep the original text
                final_dict['sentence'].append('')
                final_dict['before'].append('')
                final_dict['after'].append('')
                continue

            sentences = nltk.sent_tokenize(example['summary'][i])
            if len(sentences) < 2:
                final_dict['sentence'].append('')
                final_dict['before'].append('')
                final_dict['after'].append('')
                continue
            
            index = random.choice(range(len(sentences) - 1))
            final_dict['sentence'].append(sentences[index])
            final_dict['before'].append(' '.join(sentences[:index]))
            final_dict['after'].append(' '.join(sentences[index+1:]))

        return final_dict

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

        dict = {
            'id': examples['id'] + examples['id'],
            'text': examples['text'] + examples['text'],
            'summary': examples['summary'] + examples['summary'],
            'factual': [True] * len(examples['id']) + [False] * len(examples['id'])
        }

        return dict
        

class SumPubMedValidator:

    PROMPT_TEMPLATE = """You are tasked to evaluate whether a sentence contradicts another. Answer YES if both sentences contradict each other and NO if they don't necessarly contradict each other.\nHere is the first sentence : {sentence1}\nHere is the second sentence : {sentence2}\n\nOnly answer with YES or NO. Do not generate any other explanation."""

    def __init__(self, processed_path: str):
        self.processed_path = processed_path

        self.load()

    def load(self):

        if self.processed_path.endswith('.csv'):
            self.dataset = Dataset.from_csv(self.processed_path)
        else:
            self.dataset = load_from_disk(self.processed_path)

    def generate_validation_prompts(self):

        def generate(x, template):
            # All sentences in sumpubmed are lowercase
            return {'prompt': [template.format(sentence1=sentence.lower(), sentence2=out.lower()) for sentence, out in zip(x['sentence'], x['output'])]}

        return self.dataset.map(generate, batched=True, fn_kwargs={'template': self.PROMPT_TEMPLATE})



