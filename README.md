# Perturbation-Aware Neural ODE (pNODE) Learns Microbiome Dynamics from Clinical Data and Predicts Gut-Borne Bloodstream Infections in Patients Receiving Cancer Treatment

Requires python 3.12

Clinical_Data_Experiments contains the clinical data portion of the paper. In_Silico_Experiments contains the in silico portion. 

## Quick start guide:
To quickly load and visualize the predictions of the trained clinical model, go to Clinical_Data_Experiments/predicting_dynamics_figures.ipynb, run the first cell to load the required models and packages, and then run the cell with the function make_prediction_plot_without_reseting(). This function will show model predictions for any given patient and starting sample.

To do the same for the in silico experiments, go to In_Silico_Experiments/make_predicted_vs_true_plots.ipynb, and run make_plot_with_errors() 
