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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# np.random.seed(12345)
# I want the final output of this file to be a validation R^2 value.

def main():
    parser = argparse.ArgumentParser(description="Process input training parameters.")
    parser.add_argument('--noise', type=float, required=True, help='Percent Gaussian Noise level (float)')
    parser.add_argument('--density', type=int, required=True, help='Density value (int), number of samples out of original 49 to be subsampled per timeline')
    parser.add_argument('--training_size', type=int, required=True, help='Training size (int), i.e. number of trajectories')
    parser.add_argument('--numspecies', type=int, default=5, help='number of species in simulation')
    parser.add_argument('--training_time', type=int, default=100, help='training time for the NODE, in seconds')
    parser.add_argument('--node_test_rsquared_goal', type=float, default=0.98, help='if the NODE test R^2 reaches this value we end training')
    parser.add_argument('--replicate', type=int, default=1, help='just a label to distinguish different experimental replicates')
    # parser.add_argument('--makepredictionplots', type=int, default=1, help='1 to make eps file plots of the trained model, 0 if not')

    args = parser.parse_args()
    
    noise = args.noise
    density = args.density
    training_size = args.training_size
    numspecies = args.numspecies
    training_time = args.training_time
    node_test_rsquared_goal = args.node_test_rsquared_goal
    replicate = args.replicate

    print(f"Noise: {noise}")
    print(f"Density: {density}")
    print(f"Training Size: {training_size}")
    print(f"Number of species: {numspecies}")

    # load ground truth coefficients, generated in separate file
    r = np.load('GLV_system_growth_rates.npy')
    A = np.load('GLV_system_interactions.npy')
    p = np.load('GLV_system_perturbations.npy')
    

    # generate data from the ground truth coefs
    df_train, df_test, df_validation, perturbations_data_dict = generate_data(r,A,p, noise, density, numspecies, training_size)
    # train glv and node models and return validation rsquared values
    glv_validation_rsquared, node_validation_rsquared = train_test_and_validate_models(df_train, df_test, df_validation, perturbations_data_dict, 
                                                                    training_time = training_time, numspecies = numspecies, node_test_rsquared_goal = node_test_rsquared_goal)
    # print(glv_validation_rsquared, node_validation_rsquared)
    # save data along with data about the run together in a numpy array
    np.save(f'./data/experimental_results/experiment_noise={noise}_density={density}_training_size={training_size}_replicate={replicate}_results.npy', np.array([training_time, replicate, noise, density, training_size, glv_validation_rsquared, node_validation_rsquared]))



def train_test_and_validate_models(df_train, df_test, df_validation, perturbations_data_dict, 
                                                training_time = 120, numspecies = 5, node_test_rsquared_goal = 0.96):
    # train GLV
    glv_model = MyModels.GLVmodel(input_size = numspecies)
    perturbations = ['p1'] # hard coded names of the perturbations (only 1 perturbation)
    glv_model.train(df_train,perturbations_data_dict, perturbations)
    glv_validation_rsquared, _ = MyModels.prediction_scatter_and_r2(glv_model, df_validation, perturbations_data_dict, input_size = numspecies)
    # print('glv test R^2:', glv_rsquared_absolute)

    # initialize NODE model
    node_model = MyModels.PureNeuralODE(input_size = numspecies).to(device)
    best_node_model = MyModels.PureNeuralODE(input_size = numspecies).to(device)
    optimizer = optim.RMSprop(node_model.parameters(), lr = 0.001)
    node_test_rsquares = []
    epoch = 1
    # NODE training loop
    # train for a set amount of time, or until the test r squared exceeds 0.95
    start_time = time.time()
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
        node_test_rsquares.append(node_rsquared_absolute)
        # save if this model has the best test performance
        if all(node_rsquared_absolute>= x for x in node_test_rsquares):
            best_node_model.load_state_dict(node_model.state_dict())
        # if we reach a good enough test R^2 we end training here 
        if node_rsquared_absolute >= node_test_rsquared_goal:
            best_node_model.load_state_dict(node_model.state_dict())
            break
        epoch += 1
    # evaluate best node model on the validation set
    node_validation_rsquared, _ = MyModels.prediction_scatter_and_r2(best_node_model, df_validation, perturbations_data_dict, input_size = numspecies)

    return glv_validation_rsquared, node_validation_rsquared

def generate_data(r,A,p, noise, density, numspecies, 
                    training_size, test_size = 100, validation_size = 100):
    # Define gLV ODE model with perturbation 
    def runGLV(IC, r, A, t_eval, perturbation_start_time, perturbation_end_time):
        """
        Integrate the gLV ODE system with a time-dependent perturbation.
        """
        def system(t, x):
            s = x[:numspecies]
            if perturbation_start_time <= t <= perturbation_end_time:
                dxdt = s * (p + r + A @ s)
            else:
                dxdt = s * (r + A @ s)
            return dxdt
        soln = solve_ivp(system, (0, t_eval[-1]), IC, t_eval=t_eval, method='RK45')
        return soln.t, soln.y.T

    # Use Latin Hypercube Sampling for initial conditions 
    # (Assumes DOEpy or similar library is imported as build)
    design_dict = {}
    for i in range(numspecies):
        design_dict[f's{i+1}'] = [0.01, 1]  # Initial abundance range for each species

    exp_design = build.space_filling_lhs(
        design_dict,
        num_samples= training_size+test_size+validation_size
    )
    X = exp_design.values  # NS x numspecies matrix of initial conditions

    #  Simulate all experiments 
    N_samples = X.shape[0]
    tspan = (0, 48)
    teval = np.linspace(0, tspan[-1], 49)
    D = np.zeros([X.shape[0]*len(teval), numspecies])

    # For DataFrame construction
    time = list(teval) * X.shape[0]
    treatment_names = [f's{i+1}' for i in range(numspecies)]
    all_treatments = []
    perturbations_data_dict = {}

    for i, x in enumerate(X):
        # Name experiments based on initial conditions
        if sum(x > 0) == 1:
            exp_name = f"mono_exp_{i+1}"
        else:
            exp_name = f"exp_{i+1}"
        for _ in range(len(teval)):
            all_treatments.append(exp_name)
        # Randomly generate perturbation timeline for each experiment
        perturbation_end = np.random.uniform(0.0, 48.)
        perturbation_start = np.random.uniform(0.0, perturbation_end)
        perturbations_data_dict[exp_name] = (perturbation_start, perturbation_end)

        # Simulate ODE
        IC = x * 0.1  # Scale initial conditions
        t, y = runGLV(IC, r, A, teval, perturbation_start, perturbation_end)
        D[i*len(teval):(i+1)*len(teval)] = y

    # Build DataFrames for output 
    title = f'Simulated_gLV_data_nspecies={numspecies}'
    unique_treatments = np.unique(all_treatments)
    df = pd.DataFrame()
    df['Experiments'] = all_treatments
    df['Time'] = time
    for j, species_name in enumerate(treatment_names):
        df[species_name] = np.clip(D[:, j], 0.00001, np.inf)

    # randomly split the data into train, test and validation sets
    # Randomly shuffle and split unique experiment labels
    experiments = df['Experiments'].unique()
    np.random.shuffle(experiments)
    train_exp = experiments[:training_size]
    test_exp = experiments[training_size:training_size+test_size]
    val_exp = experiments[training_size+test_size:training_size+test_size+validation_size]

    # Subset the dataframe
    df_train = df[df['Experiments'].isin(train_exp)]
    df_test = df[df['Experiments'].isin(test_exp)]
    df_validation = df[df['Experiments'].isin(val_exp)]

    # randomly subsample from the training data to create irregular sampling
    df_train = df_train.groupby('Experiments', group_keys=False).sample(n=density)
    # add random gaussian noise to the training data
    df_train[treatment_names] += df_train[treatment_names] * np.clip(np.random.normal(loc=0, scale=noise, size=df_train[treatment_names].shape), 0.00001, np.inf)

    # sort just to make sure everything is in order
    df_train = df_train.sort_values(by=['Experiments', 'Time'])
    df_test = df_test.sort_values(by=['Experiments','Time'])
    df_validation = df_validation.sort_values(by=['Experiments','Time'])

    return df_train, df_test, df_validation, perturbations_data_dict

if __name__ == "__main__":
    main()
