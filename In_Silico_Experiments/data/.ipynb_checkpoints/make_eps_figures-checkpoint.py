
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import stats

# Create output directory
os.makedirs('./figure_eps_files', exist_ok=True)

# Load and combine data
npy_files = [f for f in os.listdir('./experimental_results/') if f.endswith('.npy')]
arrays = [np.load('./experimental_results/'+f) for f in npy_files]
combined_array = np.vstack(arrays)
df = pd.DataFrame(combined_array, columns=["training_time", "replicate", "noise", "density", "training_size", "glv_validation_rsquared", "node_validation_rsquared"])

def make_line_plot(data, x_col, title, xlabel, filename, hue_col=None):
    """Helper function to create line plots with error bars"""
    if hue_col:
        summary = data.groupby([x_col, hue_col]).agg({
            'glv_validation_rsquared': ['mean', 'std'],
            'node_validation_rsquared': ['mean', 'std']
        }).reset_index()
        summary.columns = [x_col, hue_col, 'glv_mean', 'glv_std', 'node_mean', 'node_std']
    else:
        summary = data.groupby(x_col).agg({
            'glv_validation_rsquared': ['mean', 'std'],
            'node_validation_rsquared': ['mean', 'std']
        }).reset_index()
        summary.columns = [x_col, 'glv_mean', 'glv_std', 'node_mean', 'node_std']
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(summary[x_col], summary['glv_mean'], yerr=summary['glv_std'],
                label='GLV', fmt='-o', capsize=4)
    plt.errorbar(summary[x_col], summary['node_mean'], yerr=summary['node_std'],
                label='Neural ODE', fmt='-s', capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel('Test $R^2$')
    plt.title(title)
    if x_col == 'noise':
        plt.gca().set_xticklabels([f'{x * 100:.0f}'+f'%' for x in plt.gca().get_xticks()])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/{filename}.eps', format='eps')
    plt.close()

def make_heatmaps(data, xlabel, ylabel, row_col, col_col, title_suffix, filename_suffix):
    """Helper function to create all three heatmaps (GLV, NODE, difference)"""
    # Check if data is empty
    if data.empty:
        print(f"Warning: No data available for {filename_suffix}")
        return
    
    # Round the values to 2 decimal places for cleaner display
    data = data.copy()
    data[row_col] = data[row_col]
    data[col_col] = data[col_col]

    # Calculate statistics using groupby.agg instead of apply
    grouped = data.groupby([row_col, col_col])
    
    # Calculate means
    glv_means = grouped['glv_validation_rsquared'].mean()
    node_means = grouped['node_validation_rsquared'].mean()
    
    # Check if we have any data after grouping
    if glv_means.empty or node_means.empty:
        print(f"Warning: No valid groups found after grouping for {filename_suffix}")
        return
    
    # Calculate differences
    differences = glv_means - node_means
    
    # Calculate p-values using apply for the t-test
    def calc_pvalue(group):
        glv_vals = group['glv_validation_rsquared'].dropna()
        node_vals = group['node_validation_rsquared'].dropna()
        
        if len(glv_vals) >= 2 and len(node_vals) >= 2:
            try:
                _, p_val = stats.ttest_ind(glv_vals, node_vals)
                return p_val
            except:
                return np.nan
        else:
            return np.nan
    
    p_values = grouped.apply(calc_pvalue)
    
    # Create pivot tables directly from the series
    pivot_glv = glv_means.unstack()
    pivot_node = node_means.unstack()
    pivot_diff = differences.unstack()
    pivot_p = p_values.unstack()
    
    # Create annotations for difference plot
    annotations = np.empty(pivot_diff.shape, dtype=object)
    for i in range(pivot_diff.shape[0]):
        for j in range(pivot_diff.shape[1]):
            diff_val = pivot_diff.iloc[i, j]
            p_val = pivot_p.iloc[i, j]
            if np.isnan(diff_val):
                annotations[i, j] = ""
            else:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                annotations[i, j] = f"{diff_val:.2f}\n{sig}" if sig else f"{diff_val:.2f}"
    
    
    # GLV heatmap
    plt.figure(figsize=(5,5))
    sns.heatmap(pivot_glv, annot=True, fmt='.2f', cmap='magma', 
                cbar_kws={'label': 'Mean GLV Test R²'})
    plt.title(f'Mean GLV Test R²')
    if row_col == 'training_size': # need to switch order of y axis if the y axis is training_size
        plt.gca().invert_yaxis()
    if row_col == 'noise':
        plt.gca().set_yticklabels([f'{y * 100:.0f}'+f'%' for y in pivot_glv.index])
    if col_col == 'noise':
        plt.gca().set_xticklabels([f'{x * 100:.0f}'+f'%' for x in pivot_glv.columns])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/glv_{filename_suffix}.eps', format='eps')
    plt.close()
    
    # NODE heatmap
    plt.figure(figsize=(5,5))
    sns.heatmap(pivot_node, annot=True, fmt='.2f', cmap='magma',
                cbar_kws={'label': 'Mean NODE Test R²'})
    plt.title(f'Mean NODE Test R²')
    if row_col == 'training_size': 
        plt.gca().invert_yaxis()
    if row_col == 'noise':
        plt.gca().set_yticklabels([f'{y * 100:.0f}' for y in pivot_glv.index])
    if col_col == 'noise':
        plt.gca().set_xticklabels([f'{x * 100:.0f}' for x in pivot_glv.columns])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/node_{filename_suffix}.eps', format='eps')
    plt.close()
    
    # Difference heatmap
    plt.figure(figsize=(5,5))
    sns.heatmap(pivot_diff, annot=annotations, cmap='RdBu_r', center=0, fmt='',
                cbar_kws={'label': 'GLV - NODE R² Difference'}, annot_kws={'fontsize': 10})
    plt.title(f'Difference in Mean R² Values (GLV - NODE)')
    if row_col == 'training_size': 
        plt.gca().invert_yaxis()
    if row_col == 'noise':
        plt.gca().set_yticklabels([f'{y * 100:.0f}' for y in pivot_glv.index])
    if col_col == 'noise':
        plt.gca().set_xticklabels([f'{x * 100:.0f}' for x in pivot_glv.columns])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./figure_eps_files/difference_{filename_suffix}.eps', format='eps')
    plt.close()

# Filter data for different experiments
df_fixed_time = df[df['training_time'] == 10800]

# Test R² vs density 
data_density = df_fixed_time[(df_fixed_time['noise'] == 0.1) & (df_fixed_time['training_size'] == 100)]
data_density.to_csv('./debugging_csv.csv')
make_line_plot(data_density, 'density', 'Test $R^2$ vs Number of Samples Per Timeline \n (fixed Noise = 10%, Number of timelines = 100)', 'Number of Samples Per Timeline', 'validation_r2_vs_density')

# Test R² vs noise at density = 3
data_noise_d3 = df_fixed_time[(df_fixed_time['density'] == 3) & (df_fixed_time['training_size'] == 100)]
make_line_plot(data_noise_d3, 'noise', 'Test $R^2$ vs Noise with 3 Samples Per Timeline \n (fixed Number of Timelines = 100)', 
               'Percent Gaussian Noise', 'validation_r2_vs_noise_density_3')
# zoom in on lower noise area 
data_noise_d3_zoomed_in = data_noise_d3[data_noise_d3['noise']<= 5]
make_line_plot(data_noise_d3_zoomed_in, 'noise', 'Test $R^2$ vs Noise with 3 Samples Per Timeline \n (fixed Number of Timelines = 100)', 
               'Percent Gaussian Noise', 'validation_r2_vs_noise_density_3_zoomed_in')

# Test R² vs noise at density = 10
data_noise_d10 = df_fixed_time[(df_fixed_time['density'] == 10) & (df_fixed_time['training_size'] == 100)]
make_line_plot(data_noise_d10, 'noise', 'Test $R^2$ vs Noise with 10 Samples Per Timeline \n (fixed Number of Timelines = 100)',
               'Percent Gaussian Noise', 'validation_r2_vs_noise_density_10')
# zoom in
data_noise_d10_zoomed_in = data_noise_d10[data_noise_d10['noise']<= 5]
make_line_plot(data_noise_d10_zoomed_in, 'noise', 'Test $R^2$ vs Noise with 10 Samples Per Timeline \n (fixed Number of Timelines = 100)',
               'Percent Gaussian Noise', 'validation_r2_vs_noise_density_10_zoomed_in')

# Test R² vs number of timelines
data_training_size = df_fixed_time[(df_fixed_time['noise'] == 0.1) & (df_fixed_time['density'] == 3)]
make_line_plot(data_training_size, 'training_size', 'Test $R^2$ vs Number of Timelines in Training Data \n (fixed Gaussian Noise = 10%, Number of Samples Per Timeline = 3)',
               'Number of Timelines in Training Set', 'validation_r2_vs_training_size')

# GRID PLOTS
# Heatmaps for noise x density (fixed training_size=100)
noises = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
densities = [2, 3, 4, 6, 8, 10]
xlabel = 'Number of Samples Per Timeline'
ylabel = 'Percent Gaussian Noise'
data_noise_density = df_fixed_time[
    (df_fixed_time['noise'].isin(noises)) & 
    (df_fixed_time['density'].isin(densities)) & 
    (df_fixed_time['training_size'] == 100)
]
make_heatmaps(data_noise_density,xlabel, ylabel, 'noise', 'density', 
              'by Noise and Number of Samples per Timeline', 'noise_x_density')

#9 Heatmaps for noise x training_size (fixed density=3)
training_sizes = [int(2**i) for i in range(8)]
training_sizes.reverse()
noises_ts = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
xlabel = 'Number of Timelines in Training Set'
ylabel = 'Percent Gaussian Noise'
data_noise_training = df_fixed_time[
    (df_fixed_time['noise'].isin(noises_ts)) & 
    (df_fixed_time['density'] == 3) & 
    (df_fixed_time['training_size'].isin(training_sizes))
]
make_heatmaps(data_noise_training,xlabel, ylabel, 'noise', 'training_size',
              'by Noise and Number of Timelines', 'noise_x_training_size')

# Heatmaps for density x training_size (fixed noise = 0.1)
noise = [0.1]
training_sizes = [int(2**i) for i in range(8)]
densities = [2,3,4,6,8,10]
xlabel = 'Number of Samples Per Timeline'
ylabel = 'Number of Timelines in Training Set'
data_density_training = df_fixed_time[
    (df_fixed_time['noise'].isin(noise)) & 
    (df_fixed_time['density'].isin(densities)) & 
    (df_fixed_time['training_size'].isin(training_sizes))
]
make_heatmaps(data_density_training,xlabel, ylabel, 'training_size', 'density', 
              'by Number of Timelines and Number of Samples Per Timeline', 'density_x_training_size')




print("All EPS figures have been generated in ./figure_eps_files/")