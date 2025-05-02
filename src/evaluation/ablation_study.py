import ast
import itertools

import pandas as pd
import matplotlib.pyplot as plt

from src.generation.templates import BASE_PROMPT_TEMPLATE
from src.models.prometheus import Prometheus
from src.evaluation.prometheus import PrometheusResultParser
from src.ontology.snomed import Snomed
from src.pipelines.ablation_pipeline import AblationPipeline


class AblationStudy:

    def generate_prometheus_prompts_all(results_df: pd.DataFrame, mimic: pd.DataFrame, snomed: Snomed, rubric: str = 'factuality'):
        """
        Given a datafram containing the results of the ablation study, this will generate the prometheus prompts to compare all variations of the method on a baseline.

        Args:
            results_df: DataFrame with the results. The columns must be the different methods (hps, hs, ...) and the rows correspond to each samples. The content of a row for each column will be the dictionary containing the extractions
            mimic: DataFrame containing the clinical notes (in column TEXT) and the ids of the notes (in the column ROW_ID)
            snomed: Snomed ontology to get the labels of the ids
            rubric: Prometheus rubric (factuality or relevance)
        """
        assert len(results_df) == len(mimic), 'Results must have the same length as the notes'
        columns = results_df.columns
        combinations = list(itertools.permutations(columns, r=2))

        samples = {
            'row_id': [],
            'a': [],
            'b': [],
            'concept_id': [],
            'concept': [],
            'a_result': [],
            'b_result': [],
            'prompt': []
        }

        for a, b in combinations:
            # for a, b in [(method_a, method_b), (method_b, method_a)]:
            for i in range(len(results_df)):
                clinical_note = mimic['TEXT'].iloc[i]
                row_id = mimic['ROW_ID'].iloc[i]
                results_a = ast.literal_eval(results_df.iloc[i][a])
                results_b = ast.literal_eval(results_df.iloc[i][b])

                all_concepts = set(results_a.keys()).union(set(results_b.keys()))

                for concept in all_concepts:
                    if concept is None or len(concept) == 0:
                        continue
                    response_a = results_a[concept] if concept in results_a else 'N/A'
                    response_b = results_b[concept] if concept in results_b else 'N/A'
                    label = snomed.get_label_from_id(str(concept))
                    prompt = Prometheus.create_prompt(
                        instruction=BASE_PROMPT_TEMPLATE.format(clinical_note=clinical_note, label=label),
                        response_a=response_a,
                        response_b=response_b,
                        rubric_type=rubric
                    )
                    samples['a'].append(a)
                    samples['b'].append(b)
                    samples['concept_id'].append(concept)
                    samples['concept'].append(label)
                    samples['a_result'].append(response_a)
                    samples['b_result'].append(response_b)
                    samples['row_id'].append(row_id)
                    samples['prompt'].append(prompt)
        return pd.DataFrame(samples)        

    @staticmethod
    def generate_prometheus_prompts(results_df: pd.DataFrame, mimic: pd.DataFrame, snomed: Snomed, baseline: str = 'normal', rubric: str = 'factuality'):
        """
        Given a datafram containing the results of the ablation study, this will generate the prometheus prompts to compare all variations of the method on a baseline.

        Args:
            results_df: DataFrame with the results. The columns must be the different methods (hps, hs, ...) and the rows correspond to each samples. The content of a row for each column will be the dictionary containing the extractions
            mimic: DataFrame containing the clinical notes (in column TEXT) and the ids of the notes (in the column ROW_ID)
            snomed: Snomed ontology to get the labels of the ids
            baseline: Baseline methods onto which all methods will be compared to
            rubric: Prometheus rubric (factuality or relevance)
        """
        assert len(results_df) == len(mimic), 'Results must have the same length as the notes'
        assert baseline in results_df, 'The baseline method must be present in the results'
        columns = results_df.columns
        combinations = [(x, baseline) for x in columns if x != baseline]

        samples = {
            'row_id': [],
            'a': [],
            'b': [],
            'concept_id': [],
            'concept': [],
            'a_result': [],
            'b_result': [],
            'prompt': []
        }

        for method_a, method_b in combinations:
            for a, b in [(method_a, method_b), (method_b, method_a)]:
                for i in range(len(results_df)):
                    clinical_note = mimic['TEXT'].iloc[i]
                    row_id = mimic['ROW_ID'].iloc[i]
                    results_a = ast.literal_eval(results_df.iloc[i][a])
                    results_b = ast.literal_eval(results_df.iloc[i][b])

                    all_concepts = set(results_a.keys()).union(set(results_b.keys()))

                    for concept in all_concepts:
                        if concept is None or len(concept) == 0:
                            continue
                        response_a = results_a[concept] if concept in results_a else 'N/A'
                        response_b = results_b[concept] if concept in results_b else 'N/A'
                        label = snomed.get_label_from_id(str(concept))
                        prompt = Prometheus.create_prompt(
                            instruction=clinical_note, #BASE_PROMPT_TEMPLATE.format(clinical_note=clinical_note, label=label),
                            response_a=response_a,
                            response_b=response_b,
                            rubric_type=rubric
                        )
                        samples['a'].append(a)
                        samples['b'].append(b)
                        samples['concept_id'].append(concept)
                        samples['concept'].append(label)
                        samples['a_result'].append(response_a)
                        samples['b_result'].append(response_b)
                        samples['row_id'].append(row_id)
                        samples['prompt'].append(prompt)
        return pd.DataFrame(samples)

    def get_results_against_baseline(results_df: pd.DataFrame, baseline: str = 'normal'):
        parser = PrometheusResultParser(data=results_df)
        win_rates = parser.calculate_win_rates()

        combinaisons = list(AblationPipeline.create_default_configs().keys())
        assert baseline in combinaisons, f'Baseline `{baseline}` not in ablation configurations : {combinaisons}'
        combinaisons.remove(baseline)
        results = {}
        for combinaison in combinaisons:
            if combinaison not in win_rates:
                continue
            win_rate = win_rates[combinaison][baseline]['effective_win_rate']
            results[combinaison] = win_rate

        return results

    def plot_data_results_against_baseline(results_df: pd.DataFrame, baseline: str = 'normal', rubric: str = 'Factuality', second_baseline: str = 'beam'):
        results = AblationStudy.get_results_against_baseline(results_df, baseline)
        return AblationStudy.plot_results_against_baseline(results, rubric=rubric, second_baseline=second_baseline)
        
    def plot_results_against_baseline(
        results, 
        baseline: str = 'normal', 
        rubric: str = 'Factuality', 
        second_baseline: str = 'beam',
        title: str = None
    ):
        score_beam = results[second_baseline]

        df = pd.DataFrame(list(results.items()), columns=['Method', 'Win Rate'])
        df = df[df['Method'] != second_baseline]
        df['Method'] = df['Method'].apply(lambda x: ' + '.join(x.upper()) if x != 'pn' else 'P*')

        def count_params(method):
            return sum([1 for param in ['H', 'P', 'S'] if param in method])

        df['Param Count'] = df['Method'].apply(count_params)

        df = df.sort_values(['Param Count', 'Win Rate'])

        df['Display Label'] = df.apply(lambda x: f"{x['Method']}", axis=1)

        plt.figure(figsize=(12, 6))

        bars = plt.barh(df['Display Label'], df['Win Rate'], height=0.4)

        plt.axvline(x=score_beam, color='red', linestyle='--', alpha=0.7, label=f'{second_baseline.capitalize()} Baseline ({score_beam}%)')

        plt.xlabel('Win Rate (%)')
        if title is None:
            title = f'Ablation Study: Win Rates Against {baseline.capitalize()} Decoding on {rubric}'
        plt.title(title)
        plt.xlim(50, max(df['Win Rate']) + 3)  # Start at 50% and give some padding on the right
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # Add the values at the end of each bar
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{df["Win Rate"].iloc[i]:.2f}%', 
                    va='center')

        plt.legend(loc='lower right')
        param_legend = "Parameters: H = Hierarchy score, P = Property score, S = Similarity score"
        plt.figtext(0.5, 0.01, param_legend, ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the parameter legend

        # Show the plot
        plt.show()
