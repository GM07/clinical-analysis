import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge_score.rouge_scorer import RougeScorer
from nltk.translate.bleu_score import sentence_bleu
# from bert_score import BERTScorer
import matplotlib.pyplot as plt
import re

from src.visualization.utils import plot_category_prediction_accuracy

class MedHalParser:
    """
    Evaluator that computes precision, recall, f1 scores on medhal dataset based on given generations. 
    It also computes the BLEU and ROUGE scores of the explanations
    """

    VALID_PATTERN = r'Factual(?::)?(?:\n| |\*)*(YES|NO)'
    EXPLANATION_PATTERN = r'### Explanation([\s\S]+)'
    YES_PATTERN = r'(?:Factual)?(?:\s|\:)(?:\[)?(?:yes|is factual)(?:\])?(?:\s|\:)?'
    NO_PATTERN = r'(?:Factual)?(?:\s|\:)(?:\[)?(?:no|not factual)(?:\])?(?:\s|\:)?'

    DATA_TO_TASK = {
        'acm': 'Information Extraction',
        'medmcqa': 'Question-Answering',
        'medqa': 'Question-Answering',
        'mednli': 'NLI',
        'sumpubmed': 'Summarization'
    }

    def __init__(self) -> None:
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        
    def load(self, dataset: str | Dataset):
        if isinstance(dataset, str):
            return Dataset.from_csv(dataset)
        else:
            return dataset

    def parse(self, dataset: str | Dataset, add_prompt: bool, output_col: str = 'OUTPUT', prompt_col: str = 'text'):
        data = self.load(dataset)
        data = data.map(
            self._get_prediction, 
            fn_kwargs={
                'add_prompt': add_prompt,
                'output_col': output_col,
                'prompt_col': prompt_col,
            },
            desc='Parsing answers'
        )
        return data       

    def evaluate(self, dataset: str | Dataset, add_prompt: bool, output_col: str = 'OUTPUT', prompt_col: str = 'text'):
        data = self.load(dataset)
        data = data.map(
            self._get_prediction, 
            fn_kwargs={
                'add_prompt': add_prompt,
                'output_col': output_col,
                'prompt_col': prompt_col,
            },
            desc='Parsing answers',
        )

        print('Total data size : ', len(data))

        filtered = data.filter(lambda x: x['valid'], desc='Filtering invalid samples')
        print('Filtered size : ', len(filtered))
        invalid = data.filter(lambda x: not x['valid'], desc='Retrieving invalid samples')
        print('Invalid size : ', len(invalid))

        y_pred = filtered['prediction']
        y_test = filtered['label']

        # We only evaluate BLEU and ROUGE if the model predicted False and that the true factual label is False
        # Otherwise, the explanation does not make sense
        valid_explanation = filtered.filter(lambda x: x['prediction'] == x['label'] and not x['label'], desc='Retrieving valid explanations')

        rouge_1, rouge_2 = self._get_rouge_scores(valid_explanation['explanation'], valid_explanation['explanation_gen'])
        bleu = self._get_bleu_scores(valid_explanation['explanation'], valid_explanation['explanation_gen'])
        # bert_score = self._get_bert_scores(valid_explanation['explanation'], valid_explanation['explanation_gen'])

        valid_explanation = valid_explanation.add_column('rouge_1', rouge_1)
        valid_explanation = valid_explanation.add_column('rouge_2', rouge_2)
        valid_explanation = valid_explanation.add_column('bleu', bleu)
        # valid_explanation = valid_explanation.add_column('bert', bert_score)


        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'rouge1': np.mean(rouge_1),
            'rouge2': np.mean(rouge_2),
            'bleu': np.mean(bleu),
            # 'bert': np.mean(bert_score).item(),
            'invalid': invalid,
            'valid': filtered,
            'valid_explanation': valid_explanation,
        }

    def show_stats(self, results):
        per_task = results['valid'].to_pandas()
        return plot_category_prediction_accuracy(per_task, category_column='source')

    def _get_prediction(self, row, add_prompt: bool, output_col: str, prompt_col: str):
        if row[output_col] is None:
            return {
                'valid': False,
                'prediction': False,
                'explanation_gen': ''
            }

        if add_prompt:
            output = row[prompt_col] + row[output_col]
        else:
            output = row[output_col]
        
        if not output:
            return {
                'valid': False,
                'prediction': False,
                'explanation_gen': ''
            }
        output = output.lower()
        matches = re.findall(self.VALID_PATTERN, output, re.IGNORECASE)

        if len(matches) > 0:
            # Format was respected
            matches_exp = re.findall(self.EXPLANATION_PATTERN, output, re.IGNORECASE)
            explanation_gen = output if len(matches_exp) == 0 else matches_exp[0]
            return {
                'valid': True,
                'prediction': matches[0].strip().lower() == 'yes',
                'explanation_gen': explanation_gen.strip()
            }

        # Format not respected, trying to extract factuality
        yes = len(re.findall(self.YES_PATTERN, output, re.IGNORECASE)) > 0
        no = len(re.findall(self.NO_PATTERN, output, re.IGNORECASE)) > 0

        return {
            'valid': yes ^ no,
            'prediction': yes,
            'explanation_gen': output
        }

    def _get_rouge_scores(self, references, predictions):
        """
        Returns the R1 and R2 scores of samples

        Args:
            references: List of ground truths (1 for each sample)
            predictions: List of predictions (1 for each sample)

        Returns:
            Tuple (R1 scores of each samples, R2 scores of each sample)
        """
        rouge_2 = []
        rouge_1 = []
        for exp, gen in zip(references, predictions):
            if exp is None or gen is None:
                continue
            rouge = self.rouge_scorer.score(exp, gen)
            rouge_1.append(rouge['rouge1'].fmeasure)
            rouge_2.append(rouge['rouge2'].fmeasure)

        return rouge_1, rouge_2

    def _get_bleu_scores(self, references, predictions):
        bleu = []
        for exp, gen in zip(references, predictions):
            if exp is None or gen is None:
                continue
            bleu.append(sentence_bleu([exp.split()], gen.split()))

        return bleu

    # def _get_bert_scores(self, references, predictions):
        # scorer = BERTScorer(model_type='/home/gmehenni/projects/def-azouaq/gmehenni/models/ModernBERT-large', device='cuda')
        # bert_scores = scorer.score(predictions, references, verbose=True, batch_size=32)
        # return bert_scores[2] # f1-measure


    @staticmethod
    def plot_model_comparison_accuracy(paths: list, model_names: list, add_prompts: list, category_column: str = 'source'):
        """
        Plots the accuracy score of multiple models across all categories with lines connecting dots.

        Args:
            paths (list): List of paths to the dataset results for each model.
            model_names (list): List of model names corresponding to the paths.
            add_prompts: List of boolean variables indicating whether for each path, the prompt must be added
            category_column (str): The column in the dataset that represents the category.
        """
        if len(paths) != len(model_names) and len(add_prompts) != len(paths):
            raise ValueError("The number of paths, model names and add_prompts variables must be the same.")

        accuracies = {}
        all_categories = set()

        for path, model_name, add_prompt in zip(paths, model_names, add_prompts):
            evaluator = MedHalParser()
            results = evaluator.evaluate(dataset=path, add_prompt=add_prompt, output_col='output')
            print(results)
            df = results['valid'].to_pandas()

            def map_categories(category):
                return MedHalParser.DATA_TO_TASK[category]

            df[category_column] = df[category_column].apply(map_categories)
            
            category_accuracies = df.groupby(category_column).apply(
                lambda x: f1_score(x['label'], x['prediction']) # sum(x['prediction'] == x['label']) / len(x) * 100
            ).to_dict()

            accuracies[model_name] = category_accuracies
            all_categories.update(category_accuracies.keys())

        all_categories = sorted(list(all_categories))
        
        # Prepare data for plotting
        model_accuracies = {model_name: [accuracies[model_name].get(category, 0) for category in all_categories] for model_name in model_names}
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        for model_name in model_names:
            plt.plot(all_categories, model_accuracies[model_name], marker='o', label=model_name)
        
        plt.xlabel('Task', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.legend()
        plt.show()
