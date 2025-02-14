from typing import Callable, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from tqdm.notebook import tqdm
tqdm.pandas()


from src.ontology.annotator import Annotator

class MimicVisualizer:
    """
    Plots different plots showing statistics about the mimic dataset. 
    """

    def __init__(self, mimic_processed_path: str):
        """
        Args:
            mimic_processed_path: Path to the mimic dataset after being processed by `mimic.Mimic`
        """
        self.data = pd.read_csv(mimic_processed_path)
        sns.set_theme('notebook')
        sns.set_style('white')

    def show_medical_diversity(self, annotator: Annotator):
        """
        Plots the medical diversity of the dataset

        Args:
            annotator: Annotator to use to annotate the dataset
        """
        df = self.data[['TEXT', 'CATEGORY']]
        df['SNOMED_IDS'] = df['TEXT'].progress_apply(lambda x: annotator.annotate(x, return_ids_only=True))
        df['WORD_COUNT'] = df['TEXT'].apply(lambda x: len(nltk.word_tokenize(x)))
        df['SNOMED_IDS_COUNT'] = df['SNOMED_IDS'].apply(len)
        df['MEDICAL_DIVERSITY'] = df['SNOMED_IDS_COUNT'] / df['WORD_COUNT']
        
        # Group by domain and calculate statistics
        domain_stats = df.groupby('CATEGORY').agg({
            'MEDICAL_DIVERSITY': ['mean', 'count']
        }).round(4)
        
        # Flatten column names
        domain_stats.columns = ['medical_diversity', 'total_samples'] 
        return domain_stats

    def filter_domain(self, min_samples: int = 20):
        """
        Removes categories that do not contain at least `min_samples` samples

        Args:
            min_samples: Minimum number of samples to be considered as a valid domain
        """
        df = self.data[['TEXT', 'CATEGORY']].copy()

        # Remove categories that do not contain at least `min_samples` samples
        category_counts = df['CATEGORY'].value_counts()
        valid_categories = category_counts[category_counts >= min_samples].index
        return df[df['CATEGORY'].isin(valid_categories)]

    def group_domains_by_sample_size(self, min_samples: int = 20, join_label_fct: Callable = None):
        """
        Groups the domains by sample size. If the number of samples is below `min_samples`,
        the domain is grouped into the "Others" category.

        Args:
            min_samples: Minimum number of samples to be considered as a valid domain
            join_label_fct: A function that takes a list of categories and returns a single
            string to label the "Others" category
        """

        # Get all categories and their counts
        all_categories = self.data['CATEGORY'].value_counts()
        
        # Split into categories meeting threshold and others
        other_categories = all_categories[all_categories < min_samples].index
        
        # Create new dataframe with others grouped
        df = self.data.copy()
        if len(other_categories) > 0:
            if join_label_fct is None:
                labels = ',\n'.join(other_categories)
                others_label = f"Others ({labels})"
            else:
                others_label = join_label_fct(other_categories)
            df = df.replace(other_categories, others_label)

        return df
    def plot_domain_distribution_pie(self, min_samples: int = 20):
        """
        Plots a pie plot of the proportion of each domain

        Args:
            min_samples: Minimum number of samples to be considered as a valid domain
        """
        df = self.group_domains_by_sample_size(min_samples)
        counts = df.value_counts()
        patches, texts, autotexts = plt.pie(
            counts, 
            # labels=counts.index,
            autopct=lambda v: f'{v:.2f}%',
            pctdistance=1.2,
            # labeldistance=1.5,
            radius=1.0
        )

        plt.setp(autotexts, size=8) # Percentage text
        plt.setp(texts, size=8) # Domain text

        plt.legend(patches, counts.index, title='Domains', bbox_to_anchor=(1.05, 1))

        # plt.title('Proportion of clinical notes associated to each category')
        return plt

    def plot_character_distribution(self):
        """
        Plots a distribution about the number of characters in clinical notes
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(x))
        return self._plot_distribution(
            text_chars, 
            'Number of characters in a clinical note', 
            'Probability', 
        )

    def plot_word_distribution(self):
        """
        Plots a distribution about the number of words in clinical notes.
        `nltk.word_tokenize` is used to separate the note into words
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(nltk.word_tokenize(x)))
        return self._plot_distribution(
            text_chars, 
            'Number of words in a clinical note', 
            'Probability', 
        )
    
    def plot_character_distribution_per_domain(self, min_samples: int = 20, single_plot: bool = True):
        """
        Creates a grid showing the smoothed distribution of characters for each domain
        with filled curves and mean lines. These distributions are estimated using KDE

        Args:
            min_samples: Minimum number of samples to be considered in the kernel density
            estimation
            single_plot: Whether to draw one single plot containing all categories or multiple
            plots per domain
        """

        join_label_fct = None
        if not single_plot:
            join_label_fct = lambda x: 'Others (' + ', '.join(x) + ')'
        df = self.group_domains_by_sample_size(min_samples, join_label_fct)

        df['COUNT'] = df['TEXT'].apply(len)
        
        if single_plot:
            return self._plot_single_distribution_per_domain(df, ('Number of characters', 'Probability'))
        
        return self._plot_multiple_distribution_per_domain(df, ('Number of characters', 'Probability'))

    def plot_word_distribution_per_domain(self, min_samples: int = 20, single_plot: bool = True):
        """
        Creates a grid showing the smoothed distribution of words for each domain
        with filled curves and mean lines. These distributions are estimated using KDE

        Args:
            min_samples: Minimum number of samples to be considered in the kernel density
            estimation
            single_plot: Whether to draw one single plot containing all categories or multiple
            plots per domain
        """

        join_label_fct = None
        if not single_plot:
            join_label_fct = lambda x: 'Others (' + ', '.join(x) + ')'
        df = self.group_domains_by_sample_size(min_samples, join_label_fct)

        df['COUNT'] = df['TEXT'].apply(lambda x: len(nltk.word_tokenize(x)))
        
        if single_plot:
            return self._plot_single_distribution_per_domain(df, ('Number of words', 'Probability'))
        
        return self._plot_multiple_distribution_per_domain(df, ('Number of words', 'Probability'))


    def _plot_distribution(self, data, xlabel: str, ylabel: str, title: str = None):
        """
        Plots a distribution of the data

        Args:
            data: The data to plot
            xlabel: The label of the x-axis
            ylabel: The label of the y-axis
            title: The title of the plot
        """
        g = sns.kdeplot(data, fill=True, alpha=0.7)
        g.set_xlim(0)
        g.set_xlabel(xlabel)
        g.set_ylabel(ylabel)
        if title:
            g.set_title(title)
        return g

    def _plot_single_distribution_per_domain(self, df, axis_labels: Tuple[str, str] = None):
        """
        Plots a single plot of the smoothed distribution for each domain
        with filled curves and mean lines. These distributions are estimated using KDE

        Args:
            df: The dataframe to plot containing the character count and domain in columns
            `COUNT` and `CATEGORY`
            axis_labels: The labels of the x-axis and y-axis
        """
        g = sns.kdeplot(
                data=df,
                x='COUNT',
                hue='CATEGORY',
                fill=True,
                alpha=0.5,
                common_norm=False,
            )
        g.get_legend().set_title('Domain')
        if axis_labels is not None:
            g.set_xlabel(axis_labels[0])
            g.set_ylabel(axis_labels[1])
        g.set_xlim(0)
        
        g.xaxis.set_major_formatter(lambda x, p: format(int(x), ','))
        
        return g

    def _plot_multiple_distribution_per_domain(self, df, axis_labels: Tuple[str, str] = None):
        """
        Plots a single plot of the smoothed distribution for each domain
        with filled curves and mean lines. These distributions are estimated using KDE

        Args:
            df: The dataframe to plot containing the character count and domain in columns
            `COUNT` and `CATEGORY`
            axis_labels: The labels of the x-axis and y-axis
        """
        g = sns.FacetGrid(
            data=df,
            col='CATEGORY',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=True,
            sharey=False
        )
        
        g.map_dataframe(
            sns.kdeplot,
            x='COUNT',
            alpha=0.7,
            fill=True,
        )
        
        for ax, category in zip(g.axes.flat, g.col_names):
            cat_data = df[df['CATEGORY'] == category]

            # Update title with count
            ax.set_title(f"{category}\n(n={len(cat_data):,} notes)")
        
        if axis_labels is not None:
            g.set_axis_labels(axis_labels[0], axis_labels[1])
        
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: format(int(x), ','))
            ax.set_xlim(0)

        return g
