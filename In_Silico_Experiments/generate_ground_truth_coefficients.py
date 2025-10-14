
import numpy as np
np.random.seed(12345)

# could change this or add it as an argparse or take user input later
numspecies = 5

# --- 1. Randomly create the parameters that define the system ---

# a_ii: self-interaction (diagonal) parameters (should be negative for stability)
mu_aii = -1.5
sigma_aii = 0.25

# a_ij: inter-species interaction parameters (off-diagonal)
mu_aij = -0.22
sigma_aij = 0.33

# Draw random diagonal (self-regulation) and off-diagonal (interaction) parameters
params_ii = np.random.normal(mu_aii, sigma_aii, numspecies)
params_ij = np.random.normal(mu_aij, sigma_aij, numspecies**2 - numspecies)

# Fill interaction matrix A
A = np.zeros((numspecies, numspecies))
k = 0
l = 0
for i in range(numspecies):
    for j in range(numspecies):
        if i == j:
            A[i, j] = -abs(params_ii[l])  # Ensure strictly negative diagonal
            l += 1
        else:
            A[i, j] = params_ij[k]
            k += 1


# --- 2. Normally distributed basal growth rates ---
mu_r = 0.36
sigma_r = 0.16
r = np.random.normal(mu_r, sigma_r, numspecies)
# Ensure all growth rates are non-negative
for k in range(len(r)):
    if r[k] < 0:
        r[k] = abs(r[k])

# --- 3. Perturbation effect ---
mu_p = -0.2*mu_r
sigma_p = 2*sigma_r
p = np.random.normal(mu_p, sigma_p, numspecies)

title = 'GLV_system'
np.save(f"{title}_growth_rates.npy", r)
np.save(f"{title}_interactions.npy", A)
np.save(f"{title}_perturbations.npy", p)