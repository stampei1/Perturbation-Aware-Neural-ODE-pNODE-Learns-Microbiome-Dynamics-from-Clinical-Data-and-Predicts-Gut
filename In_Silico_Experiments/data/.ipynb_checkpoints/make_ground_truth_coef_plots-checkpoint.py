import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt


r = np.load('../GLV_system_growth_rates.npy')
A = np.load('../GLV_system_interactions.npy')
p = np.load('../GLV_system_perturbations.npy')


def plot_coefficients(r, A, p, numspecies=5):

    # Growth Rate Visualization - tall and skinny
    fig, ax = plt.subplots(figsize=(3, 6))  # 3x taller than wide
    ax.barh(range(numspecies), r, color='red', align='center')
    ax.set_yticks(range(numspecies), labels=[f's{i+1}' for i in reversed(range(numspecies))])
    ax.set_xlabel('Growth Rate', fontsize=14, fontweight='bold')
    
    # Make tick labels larger and bold
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/growth_coefs.eps', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Perturbation Susceptibility - tall and skinny
    fig, ax = plt.subplots(figsize=(3, 6))  # 3x taller than wide
    colors = ['red' if val >=0 else 'blue' for val in p]  # Red=benefit, Blue=harm
    ax.barh(range(numspecies), p, color=colors, align='center')
    ax.set_yticks(range(numspecies), labels=[f's{i+1}' for i in reversed(range(numspecies))])
    
    # Make tick labels larger and bold
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/perturbation_coefs.eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Interaction Matrix Heatmap - keep square but improve labels
    plt.figure(figsize=(6, 6))
    sns.heatmap(A, cmap='coolwarm', center=0,
                xticklabels=[f's{i+1}' for i in range(numspecies)],
                yticklabels=[f's{i+1}' for i in range(numspecies)])
    
    # Make tick labels larger and bold
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/interaction_coefs.eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

plot_coefficients(r, A, p)