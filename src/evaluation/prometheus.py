import re
import itertools
from typing import List, Tuple
import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from nltk.translate import bleu_score
import pandas as pd
from tqdm import tqdm

from src.data.dataset import Dataset
from src.data.dataset import Dataset, ComparisonExtractionDataset
from src.generation.ontology_prompter import OntologyPrompter
from src.generation.templates import BASE_PROMPT_TEMPLATE
from src.models.prometheus import Prometheus
from src.ontology.snomed import Snomed

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

    def get_model_prompt(self, clinical_note, concept_id):
        """Reproduces the prompt used in the extraction step with a clinical note and an ontology concept"""
        prompter = OntologyPrompter(snomed=self.snomed, constrained_model=None)
        return prompter.create_prompts(clinical_note=clinical_note, concept_ids=[concept_id])[0]

    def get_prometheus_prompt(self, clinical_note: str, concept_id: str, response_a: str, response_b: str, rubric: str):
        """
        Returns the prompt used by prometheus to evaluate the responses
        
        Args:
            clinical_note: Clinical note used in the extraction step
            concept_id: Ontology concept id used in the extraction step
            response_a: Response of model A
            response_b: Response of model B
            rubric: Rubric used to evaluate the responses
        """
        return Prometheus.create_prompt(
            instruction=self.get_model_prompt(clinical_note, concept_id),
            response_a=response_a,
            response_b=response_b,
            rubric_type=rubric
        )

    def generate_prompts(
        self, 
        extraction_dataset: ComparisonExtractionDataset, 
        rubric: str,
        output_file_path: str | None = None,
    ):
        """
        Will generate a dataset of all combination of answers of extracted dataset

        Args:
            extraction_dataset: DataF
            rubric: Rubric used to evaluate each sample
            output_file_path: Where to save the prompts
        """
        clinical_note_column = ComparisonExtractionDataset.CLINICAL_NOTE_COLUMN
        clinical_note_id_column = ComparisonExtractionDataset.CLINICAL_NOTE_ID_COLUMN
        result_columns = ComparisonExtractionDataset.RESULT_COLUMNS

        combinations = list(itertools.permutations(result_columns, r=2))
        results = extraction_dataset.data

        prompts = []
        for combination in combinations:
            for i, row in results.iterrows():
                clinical_note = row[clinical_note_column]
                clinical_note_id = row[clinical_note_id_column]

                if type(row[combination[0]]) == float or type(row[combination[1]]) == float:
                    continue

                a_results = row[combination[0]]
                b_results = row[combination[1]]

                assert len(a_results) == len(b_results), f'The number of retrieved concepts between A {len(a_results)} is different from B ({len(b_results)})'
                for (a_concept, a_result), (b_concept, b_result) in zip(a_results.items(), b_results.items()):
                    assert a_concept == b_concept, f'Found different concepts between the results {a_concept} vs {b_concept} at row {clinical_note_id}'
                    if len(a_concept) == 0:
                        continue

                    concept_label = self.snomed.get_label_from_id(a_concept)
                    prompt = self.get_prometheus_prompt(clinical_note, a_concept, a_result, b_result, rubric)
                    prompts.append((clinical_note_id, combination[0], combination[1], clinical_note, concept_label, a_result, b_result, prompt))
                    
        prompts = pd.DataFrame(prompts, columns=[clinical_note_id_column, 'a', 'b', clinical_note_column, 'concept', 'a_result', 'b_result', 'prompt'])
        
        if output_file_path:
            prompts.to_csv(output_file_path, index=False)
        return prompts

    def generate_prompts_mixed_results(
        self, 
        first: ComparisonExtractionDataset, 
        second: ComparisonExtractionDataset,
        methods_to_compare: List[Tuple[str, str]],
        rubric: str,
        dataset_names: Tuple[str, str],
        output_file_path: str | None = None,
    ):
        """
        Same as `generate_prompts`, but can be used to compare generations from multiple datasets (for example, if we want to compare the constrained method of model A vs beam of model B)
        """
        clinical_note_column = ComparisonExtractionDataset.CLINICAL_NOTE_COLUMN
        clinical_note_id_column = ComparisonExtractionDataset.CLINICAL_NOTE_ID_COLUMN
        for method_a, method_b in methods_to_compare:
            assert method_a in ComparisonExtractionDataset.RESULT_COLUMNS, f'Method {method_a} is not a valid comparison extraction dataset column'
            assert method_b in ComparisonExtractionDataset.RESULT_COLUMNS, f'Method {method_b} is not a valid comparison extraction dataset column'

        results_a = first.data
        results_b = second.data

        assert len(results_a) == len(results_b), f'The size of dataset A ({len(results_a)}) must be equal to the size of dataset B ({len(results_b)})'

        prompts = []
        for combination in methods_to_compare:
            datasets = [(results_a, results_b), (results_b, results_a)]
            # combinations = [(no_permutation[0], no_permutation[1]), (no_permutation[1], no_permutation[0])]
            names_permutted = [(dataset_names[0], dataset_names[1]), (dataset_names[1], dataset_names[0])]
            for names, dataset in zip(names_permutted, datasets): # We compare both (a vs b) and (b vs a), because of prometheus' consistency problem
                for i in range(len(results_a)):
                    
                    row_a = dataset[0].iloc[i]
                    row_b = dataset[1].iloc[i]

                    clinical_note = row_a[clinical_note_column]
                    clinical_note_id_a = row_a[clinical_note_id_column]
                    clinical_note_id_b = row_b[clinical_note_id_column]

                    assert clinical_note_id_a == clinical_note_id_b, f'Mismatch between the rows of A and B: Found row in A with id {clinical_note_id_a} and row in B with id {clinical_note_id_b} for index {i}. Row ids must be the same for all samples'

                    if type(row_a[combination[0]]) == float or type(row_b[combination[1]]) == float:
                        continue

                    a_results = row_a[combination[0]]
                    b_results = row_b[combination[1]]

                    concepts_a = set(a_results.keys())
                    concepts_b = set(b_results.keys())
                    for concept_id in concepts_a.intersection(concepts_b):
                        a_result = a_results[concept_id]
                        b_result = b_results[concept_id]

                        if len(concept_id) == 0:
                            continue

                        concept_label = self.snomed.get_label_from_id(concept_id)
                        prompt = self.get_prometheus_prompt(clinical_note, concept_id, a_result, b_result, rubric)
                        prompts.append((clinical_note_id_a,  f'{names[0]}_{combination[0]}', f'{names[1]}_{combination[1]}', clinical_note, concept_label, a_result, b_result, prompt))
                        
        prompts = pd.DataFrame(prompts, columns=[clinical_note_id_column, 'a', 'b', clinical_note_column, 'concept', 'a_result', 'b_result', 'prompt'])

        if output_file_path:
            prompts.to_csv(output_file_path, index=False)
        return prompts

class PrometheusResultDataset(Dataset):
    """
    Dataset format used to store prometheus results
    """
    REQUIRED_COLUMNS = ['a', 'b', 'concept', 'a_result', 'b_result', 'result']
    
    def __init__(self, dataset_path: str = None, data: pd.DataFrame = None):
        super().__init__(dataset_path=dataset_path, data=data)
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

    def __init__(self, dataset_path: str = None, data: pd.DataFrame = None):
        """
        `dataset_path` or `data` must be provided. However, only one of them must be provided

        Args:
            dataset_path: Path to dataset (csv)
            data: DataFrame containing the data
        """
        self.dataset = PrometheusResultDataset(dataset_path=dataset_path, data=data)
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

    def heatmap_view(self, effective_matches: bool = True, method_names: List[str] = None, title: str = None, custom_sort_fn = None, custom_name_fn = None):
        """
        Create a heatmap visualization of win rates between algorithms.
        
        Args:
            effective_matches: Whether to include the parsing errors and ties when calculating the win rates
            method_names: Names of the methods
            title: title of the figure
            custom_sort_fn: Sort function to sort the methods
            custom_name_fn: Function that generates the display names from the method names
        """
        data = self.calculate_win_rates()
        
        if method_names is not None:
            assert len(method_names) == len(list(data.keys())), 'The names of the methods must be equal length to the number of methods'

        # Sort algorithms by name length and then by performance
        algorithms = list(data.keys())
        
        # Calculate performance metrics (number of wins against other algorithms)
        algorithm_performance = {}
        for algo in algorithms:
            wins_count = 0
            for opponent in algorithms:
                if algo != opponent:
                    win_rate = data[algo][opponent]['effective_win_rate'] if effective_matches else data[algo][opponent]['win_rate']
                    if win_rate > 50:
                        wins_count += 1
            algorithm_performance[algo] = wins_count
        
        # Sort algorithms
        if custom_sort_fn:
            algorithms.sort(key=lambda algo: custom_sort_fn(algo, algorithm_performance[algo]))
        
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
        
        # Format the text for the heatmap cells
        text = [[f'{info_val[0]:.1f}% ' if info_val[0] != 0 else '' for val, info_val in zip(row, info_row)] for row, info_row in zip(matrix, info_matrix)]
        
        # Dynamic coloring based on win rates
        # Create a diverging color scale: red for <50%, white for 50%, green for >50%
        colorscale = [
            [0.0, 'rgb(220, 220, 220)'], # White
            [0.1, '#C62828'],  # Dark red
            [0.499999, '#E57373'],# Light red 
            [0.50, 'rgb(100, 200, 100)'], # Light green
            [1.0, 'rgb(40, 80, 40)'] # Dark green
        ]
            
        
        # Convert method names to uppercase
        display_names = method_names
        if not display_names:
            display_names = [custom_name_fn(algo) for algo in algorithms] if custom_name_fn else algorithms
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=display_names,
            y=display_names,
            text=text,
            texttemplate='%{text}',
            zmax=100,
            zmin=0,
            colorscale=colorscale,
            textfont=dict(
                color='black'
            ),
            zmid=50,  # Set the midpoint of the color scale to 50%
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
            # yaxis=dict(tickangle=45),  # Rotate y-axis labels 45 degrees
            xaxis=dict(tickangle=90)   # Rotate x-axis labels 45 degrees
        )
        return fig
