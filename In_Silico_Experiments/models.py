import os
import argparse
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.linear_model import RidgeCV

# Select the device for pytorch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# ------------------ Classes ------------------

class PureNeuralODE(nn.Module):
    """
    Neural ODE model for simulating time evolution of a system with perturbations.

    Args:
        input_size (int): Number of species (dimension of the system).
        initial_sd (float): Standard deviation for weight initialization.

    Attributes:
        net (nn.Sequential): Feedforward neural network modeling the ODE's right-hand side.
        perturbation_start (float): Start time of perturbation interval.
        perturbation_end (float): End time of perturbation interval.
    """
    def __init__(self, input_size = 5, initial_sd = 0.01):
        super(PureNeuralODE, self).__init__()
        self.input_size = input_size
        
        # Feedforward neural network: input is state + 1-dim perturbation encoding
        self.net = nn.Sequential(
            nn.Linear(input_size + 1, 20),    # input_size + 1 for perturbation
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, input_size)      
        )
        
        # Initialize network parameters
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=initial_sd)
                nn.init.constant_(m.bias, val=0)
        
        # Perturbation interval, set externally before prediction
        self.perturbation_start = None
        self.perturbation_end = None
    
    def forward(self, t, x):
        """
        Compute the derivative at time t and state x.

        Args:
            t (float): Current time.
            x (Tensor): Current state vector.

        Returns:
            Tensor: Derivative dx/dt at time t.
        """
        # Determine if perturbation is active at time t
        if self.perturbation_start <= t and t <= self.perturbation_end:
            perturbation_encoding_at_time_t = torch.ones(1, dtype = torch.float32).to(device)
        else:
            perturbation_encoding_at_time_t = torch.zeros(1, dtype = torch.float32).to(device)
        x = x.to(torch.float32).reshape(self.input_size)
        # Concatenate perturbation encoding to state vector
        x_and_perturbation_encoding = torch.cat((x, perturbation_encoding_at_time_t))
        # Pass through neural network
        return self.net(x_and_perturbation_encoding)

class GLVmodel(nn.Module):
    """
    Generalized Lotka-Volterra (GLV) model with perturbation support.

    Args:
        input_size (int): Number of species (dimension of the system).

    Attributes:
        perturbation_start (float): Start time of perturbation interval.
        perturbation_end (float): End time of perturbation interval.
        abundance_coefs (Tensor): Coefficients for species interactions.
        perturbation_coefs (Tensor): Coefficients for perturbation effects.
        growth_rates (Tensor): Intrinsic growth rates for each species.
    """
    def __init__(self, input_size = 5):
        super(GLVmodel, self).__init__()
        self.input_size = input_size
        self.perturbation_start = None
        self.perturbation_end = None
    
    def train(self, train_df, perturbations_data_dict, perturbations = ['p1']):
        """
        Fit GLV model parameters using training data. We use ridge regression. 

        Args:
            train_df (DataFrame): Training data containing time series for each experiment.
            perturbations_data_dict (dict): Mapping from experiment name to perturbation interval.
            perturbations (list): List of perturbation names.

        Returns:
            None
        """
        # Construct dataframe of abundance deltas and features for regression
        df_deltas = pd.DataFrame(
            columns = ['delta_log_'+species for species in train_df.columns[2:]] +
                      [species + '_start' for species in train_df.columns[2:]] +
                      [species + '_stop' for species in train_df.columns[2:]] +
                      perturbations + ['start_time', 'stop_time', 'Experiment']
        )

        for experiment in train_df['Experiments'].unique():
            df_experiment = train_df[train_df['Experiments']==experiment]
            perturbation_interval = perturbations_data_dict[experiment]

            for i in range(len(df_experiment)-1):
                start_data = df_experiment.iloc[i, :]
                stop_data = df_experiment.iloc[i+1, :]
                start_time = start_data['Time']
                stop_time = stop_data['Time']
                start_abundances = np.array(start_data[2:])
                stop_abundances = np.array(stop_data[2:])
                # Compute per-time log change
                delta = np.log((stop_abundances/start_abundances).astype(np.float32))/(stop_time-start_time)
                # Determine if perturbation overlaps interval
                overlap = int(((perturbation_interval[0] <= stop_time) & 
                               (perturbation_interval[1] > start_time)))
                perturbation_encoding = [overlap]
                row = list(delta) + list(start_abundances) + list(stop_abundances) + perturbation_encoding + [start_time] + [stop_time] + [experiment] 
                df_deltas.loc[len(df_deltas)] = row
        df_deltas = df_deltas.dropna()
        
        # Prepare regression targets and features
        Y = df_deltas.iloc[:,:self.input_size]
        X = pd.concat([
            df_deltas.iloc[:, self.input_size:2*self.input_size],  # start abundances
            df_deltas.iloc[:, 3*self.input_size:3*self.input_size+1]  # perturbation encoding
        ], axis = 1)
        # Fit Ridge regression to estimate parameters
        model = RidgeCV()
        model.fit(X, Y)
        self.abundance_coefs = torch.from_numpy(model.coef_[:,:self.input_size]).to(device).to(torch.float32)
        self.perturbation_coefs = torch.from_numpy(model.coef_[:,self.input_size:]).to(device).to(torch.float32)
        self.growth_rates = torch.from_numpy(model.intercept_).to(device).to(torch.float32)
        return None

    def abundances_function(self, x):
        """
        Compute growth + interaction term for abundances.

        Args:
            x (Tensor): Current state vector.

        Returns:
            Tensor: Growth rates plus species interactions.
        """
        return self.growth_rates + torch.matmul(self.abundance_coefs, x)
    
    def perturbation_function(self, t, x):
        """
        Compute the perturbation effect at time t.

        Args:
            t (float): Current time.
            x (Tensor): Current state vector.

        Returns:
            Tensor: Perturbation effect vector.
        """
        if (self.perturbation_start <= t) and (t <= self.perturbation_end):
            return self.perturbation_coefs.to(device).reshape(self.input_size)
        else:
            return torch.zeros_like(self.perturbation_coefs, dtype = torch.float32).to(device).reshape(self.input_size)
    
    def forward(self, t, x):
        """
        Compute the derivative at time t and state x.

        Args:
            t (float): Current time.
            x (Tensor): Current state vector.

        Returns:
            Tensor: Derivative dx/dt at time t.
        """
        return torch.matmul(torch.diag(x), self.perturbation_function(t, x) + self.abundances_function(x)).to(device)

# ------------------ Functions ------------------

def get_data(experiment_name, model, df_data, perturbations_data_dict):
    """
    Make predictions and prepare time, initial state, and true values for a single experiment. This works for both the NODE and GLV models. 

    Args:
        experiment_name (str): Name of the experiment.
        model (nn.Module): Model to use for prediction.
        df_data (DataFrame): Full dataset. The experiment data should be in this dataset. 
        perturbations_data_dict (dict): Mapping from experiment name to perturbation interval.

    Returns:
        t (Tensor): Time points.
        y0 (Tensor): Initial state.
        y_preds (Tensor): Model predictions over time.
        y_true (Tensor): True values over time.
    """
    # set perturbation data in the model
    model.perturbation_start, model.perturbation_end = perturbations_data_dict[experiment_name]
    # get data for the experiment
    df_experiment = df_data[df_data['Experiments']==experiment_name]
    t = torch.from_numpy(df_experiment['Time'].values).to(device)
    y0 = torch.from_numpy((df_experiment.iloc[0,2:].values).astype(np.float32)).to(device)
    # predict using ODE solver
    y_preds = odeint(model, y0, t).to(device)
    # get true values for the experiment
    y_true = torch.from_numpy((df_experiment.iloc[:,2:].values).astype(np.float32)).to(device)
    return t, y0, y_preds, y_true

def get_batch_data(list_of_experiment_names, model, df_data, perturbations_data_dict):
    """
    Get predictions and true values for a batch of experiments.

    Args:
        list_of_experiment_names (list): List of experiment names.
        model (nn.Module): Model to use for prediction.
        df_data (DataFrame): Full dataset.
        perturbations_data_dict (dict): Mapping from experiment name to perturbation interval.

    Returns:
        batch_y_preds (list): List of predicted values for each experiment.
        batch_y_true (list): List of true values for each experiment.
    """
    batch_y_preds = []
    batch_y_true = []
    for experiment_name in list_of_experiment_names:
        t, y0, y_preds, y_true = get_data(experiment_name, model, df_data, perturbations_data_dict)
        batch_y_preds.append(y_preds)
        batch_y_true.append(y_true)
    return batch_y_preds, batch_y_true

def loss(batch_y_preds, batch_y_true, input_size = 5):
    """
    Compute the mean absolute error loss for a batch, handling variable sequence lengths.

    Args:
        batch_y_preds (list): List of predicted sequences (Tensors).
        batch_y_true (list): List of true sequences (Tensors).
        input_size (int): Number of species.

    Returns:
        loss (Tensor): Mean absolute error over all valid (non-padded) entries.
    """
    padded_group1 = pad_sequence(batch_y_preds, batch_first=True)
    padded_group2 = pad_sequence(batch_y_true, batch_first=True)

    max_seq_len = max(padded_group1.shape[1], padded_group2.shape[1])
    # Pad to same length along sequence dimension
    padded_group1 = F.pad(padded_group1, (0, 0, 0, max_seq_len - padded_group1.shape[1], 0, 0))
    padded_group2 = F.pad(padded_group2, (0, 0, 0, max_seq_len - padded_group2.shape[1], 0, 0))

    # Mask for valid (non-padded) values
    mask = (padded_group1 != 0) & (padded_group2 != 0)
    differences = torch.abs(padded_group1 - padded_group2)
    masked_differences = differences * mask
    loss = masked_differences.sum() / mask.sum()
    return loss

def make_batches(list_of_experiments, batch_size = 10):
    """
    Shuffle and split a list of experiment names into batches.

    Args:
        list_of_experiments (list): Experiment names.
        batch_size (int): Number of experiments per batch.

    Returns:
        list: List of batches (each a list of experiment names).
    """
    list_of_experiments = list(list_of_experiments)
    random.shuffle(list_of_experiments)
    return [list_of_experiments[i:i + batch_size] for i in range(0, len(list_of_experiments), batch_size)]

def prediction_scatter_and_r2(model, test_df, perturbations_data_dict, input_size = 5, title = 'Abundance predictions on held out test data'):
    """
    Plot scatter plots and compute R^2 for absolute and relative abundances on test data.

    Args:
        model (nn.Module): Trained model for prediction.
        test_df (DataFrame): Test data.
        perturbations_data_dict (dict): Mapping from experiment name to perturbation interval.
        input_size (int): Number of species.
        title (str): Plot title.

    Returns:
        rsquared_absolute (float): R^2 for absolute abundances.
        rsquared_relative (float): R^2 for relative abundances.
    """
    test_experiments = list(test_df['Experiments'].unique())
    test_y_preds, test_y_true = get_batch_data(test_experiments, model, test_df, perturbations_data_dict)
    predictions_np, trues_np = np.zeros(shape = (input_size,0)), np.zeros(shape = (input_size,0))
    for i in range(len(test_y_preds)):
        prediction_np = test_y_preds[i].cpu().detach().numpy().transpose()
        true_np = test_y_true[i].cpu().detach().numpy().transpose()
        # Remove first observation (initial condition)
        prediction_np = prediction_np[:, 1:]
        true_np = true_np[:, 1:]
        predictions_np = np.append(predictions_np, prediction_np, axis = 1)
        trues_np = np.append(trues_np, true_np, axis = 1)
    
    # Absolute abundances scatter plot and R^2
    x = trues_np.flatten()
    y = predictions_np.flatten()
    # plt.figure(figsize = (10,8))
    # plt.xlabel('True Abundance')
    # plt.ylabel('Predicted Abundance')
    # plt.title(title)
    # plt.scatter(x, y, color = 'red', s = 5, alpha = 0.05)
    slope, intercept, rvalue, pvalue, stddev = stats.linregress(x, y)
    # plt.plot(x, slope*x+intercept, alpha = 0.5, color = 'black')
    # plt.plot(x, x, linestyle = 'dotted', color = 'blue')
    # plt.legend([f'R^2 = {round(rvalue**2,3)}', f'm = {round(slope,3)}'])
    # plt.show()
    rsquared_absolute = rvalue**2
    
    # Relative abundances scatter plot and R^2
    predictions_np[predictions_np<0] = 0
    predictions_np = predictions_np/predictions_np.sum(axis=0, keepdims = True)
    trues_np = trues_np/trues_np.sum(axis = 0, keepdims = True)
    x = trues_np.flatten()
    y = predictions_np.flatten()
    # plt.figure(figsize = (10,8))
    # plt.xlabel('True Relative Abundance')
    # plt.ylabel('Predicted Relative Abundance')
    # plt.title(title)
    # plt.scatter(x, y, color = 'red', s = 5, alpha = 0.05)
    slope, intercept, rvalue, pvalue, stddev = stats.linregress(x, y)
    # plt.plot(x, slope*x+intercept, alpha = 0.5, color = 'black')
    # plt.plot(x, x, linestyle = 'dotted', color = 'blue')
    # plt.legend([f'R^2 = {round(rvalue**2,3)}', f'm = {round(slope,3)}'])
    # plt.show()
    rsquared_relative = rvalue**2
    
    return rsquared_absolute, rsquared_relative

    
    

    
    
    
def visualize_prediction(model, experiment_name, df):
    """
    Plots time series bar plot of model predictions. Works for both GLV and NODE

    Args:
        model (nn.Module): Trained model for prediction.
        experiment_name (str): Name of experiment.
        df (DataFrame): Dataframe with data from the experiment we want to predict and plot

    Returns:
        None
    """
    t, y0, y_preds, y_true = MyModels.get_data(experiment_name, model, df, perturbations_data_dict)
    y_preds = y_preds.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    time_points = t.cpu().detach().numpy()
    perturbation_times = perturbations_data_dict[experiment_name]
    def plot(data, time_points, title = 'Predicted trajectory'):
        row_sums = data.sum(axis=1, keepdims=True)
        relative_abundances = data / row_sums

        # Create the stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up the bottom position for each bar segment
        bottom = np.zeros(len(time_points))

        # Define nice colors for the species
        colors = [plt.cm.tab20(i / model.input_size) for i in range(model.input_size)]

        # Plot each species
        for i in range(model.input_size):
            species_data = relative_abundances[:, i]
            ax.bar(time_points, species_data, bottom=bottom, width=0.8, 
                   label=f'Species {i+1}', color=colors[i])
            bottom += species_data
            
        ax.axvspan(perturbation_times[0], perturbation_times[1], alpha=0.2, color='red', label = 'Perturbation')

        
        # Add labels and legend
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Relative Abundance', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')

        # Set axis limits
        ax.set_xlim(0, 49.5)
        ax.set_ylim(0, 1.0)

        # Add grid lines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
        return None
    plot(y_preds, time_points)
    plot(y_true, time_points,title = 'True Trajectory')
    return None
    
    
    
    