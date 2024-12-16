import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_winrate(results):
    """
    Plots head to head win rates between methods.

    Args:
        results: Dictionary where the keys are the methods and the values are another
        dictionary giving the win rate against each method as well as the number of matchups
        
    """
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
    plt.title('Head-to-Head Performance (excluding ties)')
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
