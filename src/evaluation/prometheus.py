import ast
from math import floor
import re
import itertools
from typing import List
import matplotlib.pyplot as plt
from src.data.dataset import Dataset
import re

import plotly.graph_objects as go
import plotly.express as px

from nltk.translate import bleu_score
import pandas as pd
from tqdm import tqdm

from src.data.dataset import Dataset, ExtractionDataset
from src.generation.templates import BASE_PROMPT_TEMPLATE
from src.ontology.snomed import Snomed

FACTUALITY_RUBRIC = f"""[Are the model's responses factually correct and well-supported by evidence?]
Score 1: The model's responses are mostly incorrect or based on unfounded information.
Score 2: The model sometimes provides factually correct responses, but inaccuracies are common.
Score 3: The model generally provides factually correct information, though some errors occur.
Score 4: The model often provides factually accurate information with only occasional minor errors.
Score 5: The model consistently provides responses that are factually correct and well-supported by evidence.""".strip()


RELEVANCE_RUBRIC = f"""[Are the model's responses relevant to the medical concept mentioned?]
Score 1: The model's answer is irrelevant to the medical concept and completely misses information that is related to the medical concept
Score 2: The model's short summary is mainly irrelevant, but mentions one or two things related to the medical concept mentioned
Score 3: The model's short summary is somewhat irrelevant, but contains key elements related to the concept mentioned
Score 4: The model's short summary is mainly relevant, but contains some elements that are not linked to the medical concept
Score 5: The model's short summary mentions everything related the the medical concept perfectly without missing any detail""".strip()

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{rubric}

###Feedback: """

class PrometheusPromptGenerator:
    """
    Evaluates extractions on clinical notes from the mimic dataset using the Prometheus model
    """

    def __init__(self, snomed_path: str, snomed_cache_path: str):
        """
        Args:
            snomed_path: Path to snomed ontology
            snomed_cache_path: Path to snomed cache
            loading_config: Configuration object on how to load the model
        """
        self.snomed_path = snomed_path
        self.snomed_cache_path = snomed_cache_path

        self.load()
    
    def load(self):
        """Loads the ontology"""
        self.snomed = Snomed(self.snomed_path, self.snomed_cache_path, nb_classes=366771)

    def get_model_prompt(self, clinical_note, ontology_concept):
        """Reproduces the prompt used in the extraction step with a clinical note and an ontology concept"""
        return BASE_PROMPT_TEMPLATE.format(clinical_note=clinical_note, label=ontology_concept)

    def get_prometheus_prompt(self, clinical_note: str, ontology_concept: str, response_a: str, response_b: str, rubric: str):
        """
        Returns the prompt used by prometheus to evaluate the responses
        
        Args:
            clinical_note: Clinical note used in the extraction step
            ontology_concept: Ontology concept used in the extraction step
            response_a: Response of model A
            response_b: Response of model B
            rubric: Rubric used to evaluate the responses
        """
        if rubric not in ['factuality', 'relevance']:
            raise ValueError(f'The provided rubric ({rubric}) is not supported. Here are the supported rubrics : factuality, relevance')
        
        if rubric == 'factuality':
            rubric = FACTUALITY_RUBRIC
        else:
            rubric = RELEVANCE_RUBRIC

        return ABS_SYSTEM_PROMPT + '\n\n' + ABSOLUTE_PROMPT.format(
            instruction=self.get_model_prompt(clinical_note, ontology_concept),
            response_A=response_a,
            response_B=response_b,
            rubric=rubric
        )

    def generate_prompts(
        self, 
        extraction_dataset: ExtractionDataset, 
        rubric: str,
        output_file_path: str,
    ):
        """
        Will generate a dataset of all combination of answers of extracted dataset

        Args:
            extraction_dataset: DataF
            rubric: Rubric used to evaluate each sample
            output_file_path: Where to save the prompts
        """
        clinical_note_column = ExtractionDataset.CLINICAL_NOTE_COLUMN
        clinical_note_id_column = ExtractionDataset.CLINICAL_NOTE_ID_COLUMN
        result_columns = ExtractionDataset.RESULT_COLUMNS

        combinations = list(itertools.permutations(result_columns, r=2))
        results = extraction_dataset.data


        prompts = []
        for combination in combinations:
            for i, row in results.iterrows():
                clinical_note = row[clinical_note_column]
                clinical_note_id = row[clinical_note_id_column]

                if type(row[combination[0]]) == float or type(row[combination[1]]) == float:
                    continue

                a_results = ast.literal_eval(row[combination[0]])[0]
                b_results = ast.literal_eval(row[combination[1]])[0]

                assert len(a_results) == len(b_results), f'The number of retrieved concepts between A {len(a_results)} is different from B ({len(b_results)})'
                for (a_concept, a_result), (b_concept, b_result) in zip(a_results.items(), b_results.items()):
                    assert a_concept == b_concept, f'Found different concepts between the results {a_concept} vs {b_concept} at row {clinical_note_id}'
                    concept_label = self.snomed.get_label_from_id(a_concept)
                    prompt = self.get_prometheus_prompt(clinical_note, concept_label, a_result, b_result, rubric)
                    prompts.append((clinical_note_id, combination[0], combination[1], clinical_note, concept_label, a_result, b_result, prompt))
                    
        prompts = pd.DataFrame(prompts, columns=[
            clinical_note_id_column,
            'a',
            'b',
            clinical_note_column,
            'concept',
            'a_result',
            'b_result',
            'prompt'])
        
        prompts.to_csv(output_file_path, index=False)
        return prompts


class PrometheusResultDataset(Dataset):
    """
    Dataset format used to store prometheus results
    """
    REQUIRED_COLUMNS = ['a', 'b', 'TEXT', 'concept', 'a_result', 'b_result', 'prompt', 'result']
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)


        self.verify()

    def verify(self):
        """Verifies that the data loaded is conform to the extraction dataset template"""
        error = f'The dataset is not a valid Prometheus result dataset, missing the column : '
        for required_column in PrometheusResultDataset.REQUIRED_COLUMNS:
            assert required_column in self.data.columns, error + required_column


class PrometheusResultParser:
    """
    Parses the result of Prometheus and calculates the win rate for each method
    """

    def __init__(self, prom_result_path: str):
        self.prom_result_path = prom_result_path
        self.dataset = PrometheusResultDataset(self.prom_result_path)
        self.decision_pattern = r'(?:\[?Result\]?:?)\s*(N\/A|A|B)\s*(or|and)?\s*(A|B)?' #r'(?:\[RESULT\]) (A|B)'
        self.tie_pattern = r'(?:\[RESULT\]) (A|B) or (A|B)'
        self.parse()

    def analyze_generations(self, data: pd.DataFrame = None):
        """
        Will analyze the generations of Prometheus giving more statistics about the wins and the losses
        """

        if data is None:
            data = self.dataset.data

        identical_generations = 0 # When the generate is identical in both methods
        no_concepts = 0 # No concepts were detected
        not_enough_tokens = 0
        correct_generations = 0 # Don't know why, should be 0
        model_na = 0 # Both models output N/A for the concept

        for a_result, b_result, values in tqdm(zip(data['a_result'], data['b_result'], data['result']), total=len(data)):
            a_result = str(a_result)
            b_result = str(b_result)

            if a_result.strip() == 'nan' or b_result.strip() == 'nan':
                model_na += 1
                continue

            if not isinstance(values, str):
                no_concepts += 1
                continue

            if bleu_score.sentence_bleu(a_result.lower().strip(), b_result.lower().strip()) >= 0.9:
                identical_generations += 1
                continue

            matches = re.findall(self.decision_pattern, values)
            if len(matches) == 0:
                not_enough_tokens += 1
            else:
                correct_generations += 1

        return {
            'identical_generations': identical_generations / len(data),
            'no_concepts_found': no_concepts / len(data),
            'not_enough_tokens': not_enough_tokens / len(data),
            'correct_generations': correct_generations / len(data),
            'model_na': model_na / len(data)
        }
    
    def analyze_ties(self):
        """
        Analyzes the generations in the case of ties        
        """
        
        df: pd.DataFrame = self.dataset.data
        ties = df[df['decision'] == 'tie']
        return self.analyze_generations(data=ties)

    def get_ties(self):
        """
        Returns the number of ties in the dataset
        """
        df: pd.DataFrame = self.dataset.data
        return df[df['decision'] == 'tie']

    def parse(self):
        """
        Parses the dataset by retrieving the chosen method. The decision is set in a new column called `decision`
        """
        decision = []
        for values in self.dataset.data['result']:

            if not isinstance(values, str):
                decision.append('parsing_error')
                continue

            matches = re.findall(self.decision_pattern, values, re.IGNORECASE)
            no_matches = len(matches) == 0
            if no_matches:
                decision.append('parsing_error')
                continue

            matches = matches[0]
            both_answers = len(matches) == 2 and matches[0] == 'A' and matches[1] == 'B'
            n_a = matches[0] == 'N/A'

            if both_answers or n_a:
                # Both methods were chosen
                decision.append('tie')
            else:
                decision.append(matches[0].lower().strip())

        self.dataset.data['decision'] = decision

    def sample_win(self, winner_method: str, loser_method: str):
        """
        Returns a random sample where the generation of `winner_method` won versus `loser_method`

        Args:
            winner_method: Method winning during comparison by Prometheus
            loser_method: Method losing during comparison by Prometheus
        """
        df: pd.DataFrame = self.dataset.data
        df = df[(df.a == winner_method) & (df.b == loser_method)]
        df = df[(df['a_result'] != df['b_result'])]
        df = df[df.decision == 'a']

        df2: pd.DataFrame = self.dataset.data
        df2 = df2[(df2.b == winner_method) & (df2.a == loser_method)]
        df2 = df2[(df2['a_result'] != df2['b_result'])]
        df2 = df2[df2.decision == 'b']

        samples = pd.concat([df, df2], axis=0)
        return samples.sample()

    def calculate_win_rates(self):
        """
        Calculate win rates between specific methods from a dataframe
        """

        # Get unique methods
        methods = set(self.dataset.data['a'].unique()) | set(self.dataset.data['b'].unique())
        
        results = {}
        for method1 in methods:
            results[method1] = {}
            for method2 in methods:
                if method1 == method2:
                    continue
                    
                # Get matches between these two methods
                matches = self.dataset.data.loc[
                    ((self.dataset.data['a'] == method1) & (self.dataset.data['b'] == method2)) |
                    ((self.dataset.data['a'] == method2) & (self.dataset.data['b'] == method1))
                ]
                
                total_matches = len(matches)
                if total_matches == 0:
                    continue
                    
                # Count wins for method1
                wins = len(matches.loc[
                    ((self.dataset.data['a'] == method1) & (self.dataset.data['decision'] == 'a')) |
                    ((self.dataset.data['b'] == method1) & (self.dataset.data['decision'] == 'b'))
                ])
                
                # Count parsing errors and ties
                ties = len(matches[matches['decision'] == 'tie'])
                parsing_errors = len(matches[matches['decision'] == 'parsing_error'])

                effective_matches = (total_matches - ties - parsing_errors)
                win_rate = (wins / total_matches) * 100
                effective_win_rate = (wins / effective_matches) * 100
                
                results[method1][method2] = {
                    'win_rate': round(win_rate, 2),
                    'effective_win_rate': round(effective_win_rate, 2),
                    'wins': wins,
                    'ties': ties,
                    'total_matches': total_matches,
                    'effective_matches': effective_matches,
                    'parsing_errors': parsing_errors
                }
                
        return results

    def plot_win_rates(self, title='Head-to-Head Performance (excluding ties)'):
        """
        Plots a graph showing the win rates of different methods evaluated using Prometheus

        Args:
            title: Title shown above the plot
        """
        results = self.calculate_win_rates()
        
        # Prepare data with specific ordering and grouping
        ordered_data = []
        groups = results.keys()
        
        for method in groups:
            opponents = [opp for opp in results[method].keys()]
            for opponent in opponents:
                stats = results[method][opponent]
                non_tie_matches = stats['total_matches'] - stats['ties']
                win_pct = (stats['wins'] / non_tie_matches * 100)
                ordered_data.append({
                    'matchup': f"{method} vs {opponent}",
                    'wins': win_pct,
                    'losses': 100 - win_pct
                })
        
        df = pd.DataFrame(ordered_data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot stacked bars with gaps between groups
        height = 0.6
        num_methods = len(groups)
        matches_per_method = len(ordered_data) // num_methods
        y_pos = []
        current_pos = 0
        
        for i in range(num_methods):
            group_positions = [current_pos + j for j in range(matches_per_method)]
            y_pos.extend(group_positions)
            current_pos += matches_per_method + 1  # Add gap between groups
        
        plt.barh(y_pos, df['wins'], height=height, color='#2ecc71', label='Wins')
        plt.barh(y_pos, df['losses'], height=height, left=df['wins'], color='#e74c3c', label='Losses')
        
        # Customize plot
        plt.yticks(y_pos, df['matchup'])
        plt.xlabel('Percentage')
        plt.title(title)
        plt.legend(loc='lower right')
        
        # Add percentage labels
        for i, pos in enumerate(y_pos):
            win_x = df['wins'].iloc[i] / 2
            plt.text(win_x, pos, f"{df['wins'].iloc[i]:.1f}%", 
                    ha='center', va='center', color='white')
            
            loss_x = df['wins'].iloc[i] + df['losses'].iloc[i] / 2
            plt.text(loss_x, pos, f"{df['losses'].iloc[i]:.1f}%", 
                    ha='center', va='center', color='white')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return plt


    def heatmap_view(self, effective_matches: bool = True, method_names: List[str] = None, title: str = None):
        """
        Create a heatmap visualization of win rates between algorithms.
        
        Args:
            data (dict): Nested dictionary containing win rates between algorithms
            effective_matches: Whether to include the parsing errors and ties when calculating the win rates
            method_names: Names of the methods
            method_names_y: Names of the methods for the y-axis
            title: title of the figure
        """
        data = self.calculate_win_rates()
        

        if method_names is not None:
            assert len(method_names) == len(list(data.keys())), 'The names of the methods must be of equal length to the number of methods'

        algorithms = list(data.keys())
        matrix = []
        info_matrix = []
        
        for algo1 in algorithms:
            info_row = []
            row = []
            for algo2 in algorithms:
                if algo1 == algo2:
                    info_row.append((0, 0, 0))
                    row.append(0)
                else:
                    total_matches = data[algo1][algo2]['effective_matches'] if effective_matches else data[algo1][algo2]['total_matches']
                    win_rate = data[algo1][algo2]['effective_win_rate'] if effective_matches else data[algo1][algo2]['win_rate']
                    info_row.append((win_rate, round(win_rate * total_matches / 100), total_matches))
                    row.append(win_rate)

            info_matrix.append(info_row)
            matrix.append(row)
        
        text = [[f'{info_val[0]:.1f}% \n ({info_val[1]} / {info_val[2]})' if info_val[0] != 0 else '-' for val, info_val in zip(row, info_row)] for row, info_row in zip(matrix, info_matrix)]
        text = [[f'{info_val[0]:.1f}% ' if info_val[0] != 0 else '-' for val, info_val in zip(row, info_row)] for row, info_row in zip(matrix, info_matrix)]
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=method_names if method_names else algorithms,
            y=method_names if method_names else algorithms,
            text=text,
            texttemplate='%{text}',
            # textfont={"size": 12},
            zmax=100,
            zmin=0,
            # colorscale='Reds',
            colorscale='YlGn', #[(0.0, 'white'), (0.01, 'rgb(201, 109, 109)'), (1.0, 'rgb(115, 201, 109)')],
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Win Rate: %{z:.1f}%<extra></extra>',
            showscale=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title='Algorithm A',
            xaxis_title='Algorithm B',
            width=500,
            height=500,
            yaxis=dict(tickangle=-90)  # Rotate y-axis labels 90 degrees
        )
        return fig
