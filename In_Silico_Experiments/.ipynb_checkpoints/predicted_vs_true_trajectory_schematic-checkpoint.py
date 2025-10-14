import numpy as np
import pandas as pd
import random
import torch
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

def make_plot(node_model):
    # now we want to get and plot the predicted vs true values on the test data
    i = random.randint(0, 100)
    experiment_name = df_test['Experiments'].iloc[i]
    t, y0, y_preds, y_true = MyModels.get_data(experiment_name, node_model, df_test, perturbations_data_dict)
    # convert to numpy
    t = t.detach().cpu().numpy()
    y_preds = y_preds.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    df_plot = pd.DataFrame({
        'Time': t,
        's1': y_true[:, 0],
        's2': y_true[:, 1], 
        's3': y_true[:, 2],
        's4': y_true[:, 3],
        's5': y_true[:, 4],
        's1 Predicted': y_preds[:, 0],
        's2 Predicted': y_preds[:, 1],
        's3 Predicted': y_preds[:, 2],
        's4 Predicted': y_preds[:, 3],
        's5 Predicted': y_preds[:, 4]
    })

    print(df_plot.head())
    # Plot the lines first
    ax = df_plot.set_index('Time')[['s1', 's2', 's3', 's4', 's5','s1 Predicted', 's2 Predicted', 's3 Predicted', 's4 Predicted', 's5 Predicted']].plot(legend=False)

    # Get the colors from the line plot
    colors = [line.get_color() for line in ax.get_lines()]

    # Plot markers for true values (circles) and predicted values (squares)
    df_indexed = df_plot.set_index('Time')
    true_columns = ['s1', 's2', 's3', 's4', 's5']
    pred_columns = ['s1 Predicted', 's2 Predicted', 's3 Predicted', 's4 Predicted', 's5 Predicted']

    # Plot true values with circles
    for i, col in enumerate(true_columns):
        ax.scatter(df_indexed.index, df_indexed[col], marker='o', color=colors[i], s=20)

    # Plot predicted values with squares (using the same colors as their corresponding true values)
    for i, col in enumerate(pred_columns):
        ax.scatter(df_indexed.index, df_indexed[col], marker='s', color=colors[i], s=20)

    start, stop = perturbations_data_dict[experiment_name]
    ax.axvspan(start, stop, color='blue')
    plt.savefig(f'./data/figure_eps_files/predicted_vs_true_timelines.eps', dpi=300, bbox_inches='tight')
    plt.show()
    return None


r = np.load('GLV_system_growth_rates.npy')
A = np.load('GLV_system_interactions.npy')
p = np.load('GLV_system_perturbations.npy')
noise = 0.0
density = 3
numspecies = 5
training_size = 100

df_train, df_test, df_validation, perturbations_data_dict = generate_data(r,A,p, noise, density, numspecies, training_size)

node_model = MyModels.PureNeuralODE(input_size = numspecies).to(device)
state_dict = torch.load('./trained_node_for_plotting_with_rsquared_0-9.pt')
node_model.load_state_dict(state_dict)
optimizer = optim.RMSprop(node_model.parameters(), lr = 0.001)
start_time = time.time()
epoch = 1
training_time = 2*3600 # 2 hours
start_time = time.time()
rsquared_checklist = []
print('starting node training')
while time.time() - start_time < training_time:
    batches = MyModels.make_batches(df_train['Experiments'].unique())
    for batch_experiments in batches:
        optimizer.zero_grad()
        batch_y_preds, batch_y_true = MyModels.get_batch_data(batch_experiments, 
                                                                node_model,
                                                                df_train,
                                                                perturbations_data_dict)
        loss = MyModels.loss(batch_y_preds, batch_y_true, input_size = numspecies)
        loss.backward()
        optimizer.step()
    node_rsquared_absolute, _ = MyModels.prediction_scatter_and_r2(node_model, df_test, perturbations_data_dict, input_size = numspecies)
    
    if (node_rsquared_absolute > 0.5) and (node_rsquared_absolute <0.6) and (0.5 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-5.pt')
        rsquared_checklist.append(0.5)
    if (node_rsquared_absolute > 0.6) and (node_rsquared_absolute <0.7) and (0.6 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-6.pt')
        rsquared_checklist.append(0.6)
    if (node_rsquared_absolute > 0.7) and (node_rsquared_absolute <0.8) and (0.7 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-7.pt')
        rsquared_checklist.append(0.7)
    if (node_rsquared_absolute > 0.8) and (node_rsquared_absolute <0.9) and (0.8 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-8.pt')
        rsquared_checklist.append(0.8)
    if (node_rsquared_absolute > 0.9) and (node_rsquared_absolute <0.95) and (0.9 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-9.pt')
        rsquared_checklist.append(0.9)
    if (node_rsquared_absolute > 0.95) and (node_rsquared_absolute <1.0) and (0.95 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-95.pt')
        rsquared_checklist.append(0.95)
    if (node_rsquared_absolute > 0.97) and (node_rsquared_absolute <1.0) and (0.97 not in rsquared_checklist):
        torch.save(node_model.state_dict(), f'./trained_node_for_plotting_with_rsquared_0-97.pt')
        rsquared_checklist.append(0.97)
    epoch += 1
print(epoch)
print('node training done. saving the model to the current directory')
torch.save(node_model.state_dict(), './trained_node_for_plotting.pt')

# node_model = MyModels.PureNeuralODE(input_size = numspecies).to(device)
# state_dict = torch.load('./trained_node_for_plotting.pt')
# node_model.load_state_dict(state_dict)
# make_plot(node_model)