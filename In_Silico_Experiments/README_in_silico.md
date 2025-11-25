generate_ground_truth_coefficients.py generates the ground truth GLV coefficiets for a series of in silico experiments by sampling from a multivariate normal, and saves them as .npy files in the same directory.

run_experiment.py loads the generated ground truth coeffients, and then runs a pNODE vs. GLV experiment according to the args it is given, including training data noise, training data sampling density (out of 48 total per trajectory), training size, which is the total number of trajectories in the training. The results of the experiemnt, which are test R^2 values, are recorded and saved as .npy files in data/experimental_results



