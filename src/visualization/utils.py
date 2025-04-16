import pandas as pd
import matplotlib.pyplot as plt


def plot_category_prediction_accuracy(df: pd.DataFrame, category_column: str = 'category', reference_column: str = 'label', prediction_column: str = 'prediction'):
    """
    Create a stacked bar plot showing prediction accuracy by category.
    
    Parameters:
    df (pandas.DataFrame): DataFrame 
    
    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    # Group by category and calculate accuracy
    accuracy_df = df.groupby(category_column).apply(
        lambda x: pd.Series({
            'total': len(x),
            'correct': sum(x[prediction_column] == x[reference_column]) / len(x) * 100,
            'incorrect': sum(x[prediction_column] != x[reference_column]) / len(x) * 100
        })
    ).reset_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create stacked bars
    plt.bar(
        accuracy_df[category_column], 
        accuracy_df['correct'], 
        color='green', 
        label='Correct Predictions'
    )
    plt.bar(
        accuracy_df[category_column], 
        accuracy_df['incorrect'], 
        bottom=accuracy_df['correct'], 
        color='red', 
        label='Incorrect Predictions'
    )
    
    # Customize the plot
    # plt.title('Prediction Accuracy by Category', fontsize=15)
    plt.xlabel(category_column, fontsize=12)
    plt.ylabel('Percentage of predictions', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels
    for i, row in accuracy_df.iterrows():
        correct = row['correct']
        incorrect = row['incorrect']
        
        # Percentage of correct predictions
        correct_pct = correct
        
        plt.text(
            i, 
            correct / 2, 
            f'{correct_pct:.1f}%\nCorrect', 
            ha='center', 
            va='center',
            color='white',
            # fontweight='bold'
        )
        
        # Percentage of incorrect predictions
        incorrect_pct = incorrect
        plt.text(
            i, 
            correct + incorrect / 2, 
            f'{incorrect_pct:.1f}%\nIncorrect', 
            ha='center', 
            va='center',
            color='white',
            # fontweight='bold'
        )
    
    plt.tight_layout()
    return plt
