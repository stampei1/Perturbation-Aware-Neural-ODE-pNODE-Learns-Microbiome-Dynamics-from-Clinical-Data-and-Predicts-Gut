# Perturbation-Aware Neural ODE (pNODE) Learns Microbiome Dynamics from Clinical Data and Predicts Gut-Borne Bloodstream Infections in Patients Receiving Cancer Treatment

Requires python 3.12

Clinical_Data_Experiments contains the clinical data portion of the paper. In_Silico_Experiments contains the in silico portion. 

## Quick start guide:
To quickly load and visualize the predictions of the trained clinical pNODE model, go to Clinical_Data_Experiments/predicting_dynamics_figures.ipynb, run the first cell to load the required models and packages, and then run the cell with the function make_prediction_plot_without_reseting(). This function will show model predictions for any given patient and starting sample.

To do the same for the in silico experiments, go to In_Silico_Experiments/make_predicted_vs_true_plots.ipynb, and run make_plot_with_errors() 


## In Silico Experiments

generate_ground_truth_coefficients.py generates the ground truth GLV coefficiets for a series of in silico experiments by sampling from a multivariate normal, and saves them as .npy files in the same directory.

run_experiment.py loads the generated ground truth coeffients, and then runs a pNODE vs. GLV experiment according to the args it is given, including training data noise, training data sampling density (out of 48 total per trajectory), training size, which is the total number of trajectories in the training. The results of the experiemnt, which are test R^2 values, are recorded and saved as .npy files in data/experimental_results.

models.py contains the code for the GLV and pNODE models, as well as functions for batching, loss, and evaluating performance. 

train_node_to_make_predicted_vs_true_plots.py trains and saves instances of the pNODE model in the current directory for different test loss values, to be used for plotting later.

make_predicted_vs_true_plots.ipynb has code to load a trained model (trained in train_node_to_make_predicted_vs_true_plots.py) and then plot model predictions against the ground truth data. 

trajectory_schematics.py makes schematics of the training data under different conditions.

make_args.py creates a text file with 1000s of combinations of training data condition arguments for run_experiment.py, in the file args.txt. 

job_array.sh is a SLURM shell script that runs copies of 'run_experiment.py' in parallel with args from args.txt. You can modify the script to run on your own cluster. 

## Clinical Data Experiments 


To visualize the predictions of the trained pNODE and GLV models, go to predicting_dynamics_figures.ipynb. To look at the infection prediction and intestinal domination prediction code go to predicting_bloodstream_infections_and_intestinal_dominations.ipynb