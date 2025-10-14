import argparse
import numpy as np
from scipy import stats
from scipy.integrate import  solve_ivp
import pandas as pd
import torch
from torchdiffeq import odeint
from doepy import build
import models as MyModels
import time
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import run_experiment
from run_experiment import generate_data

noise = 0.5
density = 3
training_size = 10
numspecies = 5

r = np.load('GLV_system_growth_rates.npy')
A = np.load('GLV_system_interactions.npy')
p = np.load('GLV_system_perturbations.npy')

df_train, df_test, df_validation, perturbations_data_dict = generate_data(r,A,p, noise, density, numspecies, training_size)

for i in range(5):
    # no noise and regular sampling
    df_plot = df_test[df_test['Experiments'] == (df_test['Experiments'].unique()[i])]
    ax = df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']].plot(legend = False)
    (df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']]).plot(style='o', ax=ax, legend = False, color=[line.get_color() for line in ax.get_lines()])
    start, stop = perturbations_data_dict[df_test['Experiments'].unique()[i]]
    ax.axvspan(start, stop,  color='blue')
    plt.savefig(f'./data/figure_eps_files/trajectory_with_no_noise_and_regular_sampling_{i}_.eps', dpi=300, bbox_inches='tight')
    plt.show()
    # noise and regular sampling
    ax = df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']].plot(legend = False)
    (df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']] + np.random.normal(0, 0.30, (len(df_plot), 5))).plot(style='o', ax=ax, legend = False, color=[line.get_color() for line in ax.get_lines()])
    start, stop = perturbations_data_dict[df_test['Experiments'].unique()[i]]
    ax.axvspan(start, stop,  color='blue') 
    plt.savefig(f'./data/figure_eps_files/trajectory_with_noise_and_regular_sampling_{i}_.eps', dpi=300, bbox_inches='tight')
    plt.show()
    # no noise, irregular sampling
    ax =df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']].plot(legend = False)
    (df_plot.sample(frac=0.08).set_index('Time')[['s1', 's2', 's3', 's4', 's5']]).plot(style='o', ax=ax, legend = False,color=[line.get_color() for line in ax.get_lines()])
    start, stop = perturbations_data_dict[df_test['Experiments'].unique()[i]]
    ax.axvspan(start, stop,  color='blue') 
    plt.savefig(f'./data/figure_eps_files/trajectory_with_no_noise_and_irregular_sampling_{i}_.eps', dpi=300, bbox_inches='tight')
    plt.show()
    # noise + irregular sampling
    ax=df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5']].plot(legend = False)
    (df_plot.sample(frac=0.08).set_index('Time')[['s1', 's2', 's3', 's4', 's5']] + np.random.normal(0, 0.30, (len(df_plot.sample(frac=0.08)), 5))).plot(style='o', ax=ax, legend = False, color=[line.get_color() for line in ax.get_lines()])
    start, stop = perturbations_data_dict[df_test['Experiments'].unique()[i]]
    ax.axvspan(start, stop,  color='blue') 
    plt.savefig(f'./data/figure_eps_files/trajectory_with_noise_and_irregular_sampling_{i}_.eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

