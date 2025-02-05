import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

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

    def filter_domain_by_min_samples(self, min_samples: int = 20):
        """
        Filters the dataset to only contain at least a certain amount of clinical notes

        Args:
            min_samples: Minimum number of samples to be considered as a valid domain
        """
        df = self.data[['TEXT', 'CATEGORY']].copy()

        # Remove categories that do not contain at least `min_samples` samples
        category_counts = df['CATEGORY'].value_counts()
        valid_categories = category_counts[category_counts >= min_samples].index
        return df[df['CATEGORY'].isin(valid_categories)]

    def plot_domain_distribution_pie(self, min_samples: int = 20):
        """
        Plots a pie plot of the proportion of each domain

        Args:
            min_samples: Minimum number of samples to be considered as a valid domain
        """
        df = self.filter_domain_by_min_samples(min_samples=min_samples)['CATEGORY']
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

        plt.title('Proportion of clinical notes associated to each category')

    def plot_character_distribution(self):
        """
        Plots a distribution about the number of characters in clinical notes
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(x))
        g = sns.kdeplot(text_chars, fill=True, alpha=0.4)
        g.set_xlim(0)
        g.set_xlabel('Number of characters in a clinical note')
        g.set_ylabel('Number of clinical notes')
        g.set_title(f'Distribution of number of characters in clinical notes (n={len(self.data)})')
        return g

    def plot_word_distribution(self):
        """
        Plots a distribution about the number of words in clinical notes.
        `nltk.word_tokenize` is used to separate the note into words
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(nltk.word_tokenize(x)))
        g = sns.kdeplot(text_chars, fill=True, alpha=0.4)
        g.set_xlim(0)
        g.set_xlabel('Number of words in a clinical note')
        g.set_ylabel('Number of clinical notes')
        g.set_title(f'Distribution of number of words in clinical notes (n={len(self.data)})')
        return g
 
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
        df = self.filter_domain_by_min_samples(min_samples)
        df['char_count'] = df['TEXT'].apply(len)
        
        if single_plot:
            g = sns.kdeplot(
                data=df,
                x='char_count',
                hue='CATEGORY',
                fill=True,
                alpha=0.2,
                common_norm=False
            )
            g.get_legend().set_title('Domain')
            g.set_xlabel('Number of characters')
            g.set_ylabel('Probability')
            g.set_ylim((0, 0.002))
            g.set_xlim(0)
            g.xaxis.set_major_formatter(lambda x, p: format(int(x), ','))
            
            return g
        
        g = sns.FacetGrid(
            data=df,
            col='CATEGORY',
            col_wrap=2,
            height=4,
            aspect=1.5,
            sharex=True,
            sharey=False
        )
        
        g.map_dataframe(
            sns.kdeplot,
            x='char_count',
            alpha=0.4,
            fill=True
        )
        
        for ax, category in zip(g.axes.flat, g.col_names):
            cat_data = df[df['CATEGORY'] == category]

            # Update title with count
            ax.set_title(f"{category}\n(n={len(cat_data):,} notes)")
        
        g.set_axis_labels('Number of characters', 'Probability')
        
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: format(int(x), ','))
            ax.set_xlim(0)

        return g
