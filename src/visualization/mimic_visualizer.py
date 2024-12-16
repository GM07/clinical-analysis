import pandas as pd
import seaborn as sns
import nltk

class MimicVisualizer:
    """
    Plots different plots showing statistics about the mimic dataset. 
    """

    def __init__(self, mimic_processed_path: str):
        """
        Args:
            mimic_processed_path: Path to the mimic dataset after being processed by `loader.MimicLoader`
        """
        self.data = pd.read_csv(mimic_processed_path)
        sns.set_theme('notebook')
        sns.set_style('white')

    def plot_character_distribution(self):
        """
        Plots a distribution about the number of characters in clinical notes
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(x))
        plt = sns.histplot(text_chars, element='poly', alpha=0.4)
        plt.set_xlabel('Number of characters in a clinical note')
        plt.set_ylabel('Number of clinical notes')
        plt.set_title('Distribution of number of characters in clinical notes')
        return plt

    def plot_word_distribution(self):
        """
        Plots a distribution about the number of words in clinical notes.
        `nltk.word_tokenize` is used to separate the note into words
        """
        text_chars = self.data['TEXT'].apply(lambda x: len(nltk.word_tokenize(x)))
        plt = sns.histplot(text_chars, element='poly', alpha=0.4)
        plt.set_xlabel('Number of words in a clinical note')
        plt.set_ylabel('Number of clinical notes')
        plt.set_title('Distribution of number of words in clinical notes')
        return plt
 
    def plot_character_distribution_per_category(self):
        """
        Creates a FacetGrid showing the smoothed distribution of characters for each category
        with filled curves and mean lines.
        """
        df = self.data[['TEXT', 'CATEGORY']].copy()
        df['char_count'] = df['TEXT'].apply(len)
        
        category_counts = df['CATEGORY'].value_counts()
        sorted_categories = category_counts.index.tolist()

        g = sns.FacetGrid(
            data=df,
            col='CATEGORY',
            col_wrap=2,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        
        g.map_dataframe(
            sns.histplot,
            x='char_count',
            fill=True,
            alpha=0.4,
            # color='blue'
            element='poly'
        )
        
        # Add mean lines and category counts
        for ax, category in zip(g.axes.flat, sorted_categories):
            # Get data for this category
            cat_data = df[df['CATEGORY'] == category]

            if category == 'Discharge summary':
                print(cat_data['char_count'].max())
            
            # Add mean line
            mean = cat_data['char_count'].mean()
            ax.axvline(mean, color='red', linestyle='--', alpha=0.7)
            ylim = ax.get_ylim()
            ax.text(mean, ylim[1], 
                   f'Mean: {int(mean):,}',
                   rotation=90,
                   va='top',
                   ha='right')
            
            # Update title with count
            ax.set_title(f"{category}\n(n={len(cat_data):,} notes)")
        
        # Label axes
        g.set_axis_labels('Number of characters', 'Number of notes')
        
        # Add thousands separator to x-axis
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: format(int(x), ','))
        
        # g.fig.tight_layout()
        return g
