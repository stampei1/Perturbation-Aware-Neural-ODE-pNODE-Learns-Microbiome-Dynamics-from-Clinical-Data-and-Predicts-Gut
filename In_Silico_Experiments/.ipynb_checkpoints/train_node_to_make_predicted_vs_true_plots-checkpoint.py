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

r = np.load('GLV_system_growth_rates.npy')
A = np.load('GLV_system_interactions.npy')
p = np.load('GLV_system_perturbations.npy')
noise = 0.0
density = 3
numspecies = 5
training_size = 100

df_train, df_test, df_validation, perturbations_data_dict = generate_data(r,A,p, noise, density, numspecies, training_size)

node_model = MyModels.PureNeuralODE(input_size = numspecies).to(device)
optimizer = optim.RMSprop(node_model.parameters(), lr = 0.001)
start_time = time.time()
epoch = 1
training_time = 3*3600 # 3 hours
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