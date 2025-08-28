import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_correlation(corr_mat: pd.DataFrame, group_sizes: list=False , group_names: list=False, size=(12, 10)) -> None:
    """
    Function to plot corr matrices.
    If group size and group_names given
    will dive correlation matric into subject groups

    Parameters
    -----------
    corr_mat: pd.DataFrame
        corr matrix
    group_sizes: list=False
        size of each group
        default is False
    group_names: list=False
        name of each group 
    size: tuple[int, int]
        size of plot
    
    Returns
    -------
    None
    """
    
    _, ax = plt.subplots(figsize=size)
    sns.heatmap(corr_mat,  xticklabels=False, yticklabels=False)
    if not group_sizes and not group_names:
        plt.show()
        return None
    
    boundaries = np.cumsum(group_sizes)    
    for bound in boundaries[:-1]:
        ax.axhline(bound, color="black", lw=2)
        ax.axvline(bound, color="black", lw=2)
    
    # Add labels at group boundaries
    start = 0
    for size, name in zip(group_sizes, group_names):
        center = start + size / 2
        ax.text(center, -0.5, name, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(-0.5, center, name, ha="right", va="center", fontsize=10, fontweight="bold", rotation=90)
        start += size
    
    
    for bound in boundaries[:-1]: 
        ax.axhline(bound, color='black', lw=2)
        ax.axvline(bound, color='black', lw=2)
    plt.show()

def off_diag_mean(matrix):
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask].mean()

def diag_mean(matrix):
    return np.diag(matrix).mean()