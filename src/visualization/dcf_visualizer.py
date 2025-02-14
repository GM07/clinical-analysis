from collections import Counter
from src.domain_adaptation.domain_analyser import DomainAnalyser
from typing import Callable, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from tqdm.notebook import tqdm
import plotly.graph_objs as go
from itertools import cycle, islice

from src.ontology.snomed import Snomed
tqdm.pandas()


class DCFVisualizer:
    """
    Visualizer for the Domain Class Frequency
    """

    def __init__(self, dcf: DomainAnalyser, snomed: Snomed):
        self.dcf = dcf
        self.snomed = snomed

        for dcf in self.dcf.domain_class_frequencies.values():
            # Make sure serialization did not break the counter
            if not isinstance(dcf.counter, Counter):
                dcf.counter = Counter(dcf.counter)

    def plot_most_frequent_concepts_per_domain(self, top_n: int = 5):
        """
        Plots the most frequent concepts per domain as a nested pie chart.
        The outer ring shows the domains, while the inner ring shows the top concepts for each domain.

        Args:
            top_n: The number of concepts to plot per domain (default 5)
        """
        # Prepare data for outer ring (domains)
        domains = list(self.dcf.domain_class_frequencies.keys())
        all_labels, ids = [], []

        colors = sns.color_palette().as_hex()
        pastel_colors = sns.color_palette('pastel').as_hex()
        domain_colors = list(islice(cycle(colors), len(domains)))
        pastel_domain_colors = list(islice(cycle(pastel_colors), len(domains)))
        concept_colors = []
        
        # Collect data for both rings
        for i, domain in enumerate(domains):
            dcf = self.dcf.domain_class_frequencies[domain]
            concepts, _ = dcf.get_concepts(top_n, separate=True)
            labels = self.snomed.convert_ids_to_labels(concepts)
            
            ids.extend(list(map(lambda x: domain + '-' + x, concepts)))
            all_labels.extend(labels)
            concept_colors.extend([pastel_domain_colors[i]] * len(concepts))
            
        # Create inner trace
        trace1 = go.Pie(
            hole=0.5,
            sort=False,
            direction='clockwise',
            domain={'x': [0.15, 0.85], 'y': [0.15, 0.85]},
            values=[1/len(domains) for _ in range(len(domains))],
            labels=domains,
            textinfo='label',
            textposition='inside',
            marker={
                'colors': domain_colors,  # Darker blue
                'line': {'color': 'black', 'width': 1}
            },
        )

        # Create outer trace
        trace2 = go.Pie(
            hole=0.7,
            sort=False,
            direction='clockwise',
            values=[1/(len(all_labels) * len(domains)) for _ in range(len(all_labels))],
            labels=ids,
            text=all_labels,
            texttemplate='%{text}',
            textinfo='label',
            textposition='inside',
            marker={
                'colors': concept_colors,  # Lighter blue
                'line': {'color': 'black', 'width': 1}
            }
        )

        # Create and return figure
        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(width=1000, height=1000, showlegend=False)
        return fig
