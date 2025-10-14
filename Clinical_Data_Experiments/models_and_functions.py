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
from sklearn.metrics import roc_curve, auc
import seaborn as sns
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

# Select the device for pytorch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    print(f"GPU Device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
    print(f"GPU Memory Cached: {round(torch.cuda.memory_cached(0)/1024**3,1)} GB")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

class PureNeuralODE(nn.Module):
    def __init__(self, df_antibiotics, antibiotic_types, input_size = 34):
        ''' Parameters:
        df_antibiotics is the antibiotic dataframe
        antibiotic_types is a list of strings giving the names of each antibiotic
        input_size is the number of taxa we will simulate on
        '''
        super(PureNeuralODE, self).__init__()
        self.input_size = input_size

        # define attribute that stores the antibiotic information
        # we will store it as a dictionary - lookup table
        # df_antibiotics should be the dataframe containing all of the antibiotic information
        self.antibiotic_types = antibiotic_types
        self.antibiotics_dict = self.preprocess_antibiotics(df_antibiotics, antibiotic_types)
        
        # current patient_id, to be used in a forward pass, and changed outside the model during training
        self.patient_id = None
        
        # feed forward neural net
        self.net = nn.Sequential(
            nn.Linear(input_size + 15 + 1, 20),    # 15 antibiotics + input_size taxa + time = 50 input
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, input_size)      
        )
        
        # intialize parameters of self.net
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
        
    def preprocess_antibiotics(self, df,antibiotic_types, time_resolution=0.1):
        categories = antibiotic_types
        antibiotics_dict = {}

        for _, row in df.iterrows():
            patient_id = row['PatientID']
            start_day = row['StartDayRelativeToNearestHCT']
            stop_day = row['StopDayRelativeToNearestHCT']
            category = row['Category']

            # Initialize dictionary for this patient if not already done
            if patient_id not in antibiotics_dict:
                antibiotics_dict[patient_id] = {}

            # Get indices for the category
            category_idx = categories.index(category)

            # Loop through the discretized time range
            times = np.arange(start_day, stop_day + time_resolution, time_resolution)
            for t in times:
                rounded_t = round(t, int(-np.log10(time_resolution)))  # Round to match resolution
                if rounded_t not in antibiotics_dict[patient_id]:
                    antibiotics_dict[patient_id][rounded_t] = [0] * len(categories)
                antibiotics_dict[patient_id][rounded_t][category_idx] = 1

        return antibiotics_dict
        
    def get_antibiotics_encodings(self, patient_id, time, time_resolution=0.1):
        # time comes in as a tensor so we need to convert it to numpy
        time = time.item()
        # note: you need to use resolution 0.1 
        rounded_time = round(time, int(-np.log10(time_resolution)))  # Round to match resolution
        if patient_id in self.antibiotics_dict and rounded_time in self.antibiotics_dict[patient_id]:
            return torch.tensor(self.antibiotics_dict[patient_id][rounded_time], dtype=torch.float32)
        else:
            return torch.zeros(len(self.antibiotic_types), dtype=torch.float32)
    
    def antibiotic_function(self, x, query_time):
        # x is the hidden state of the model, ie the diversity, and is a tensor of shape (1)
        # query_time is the time that this function is being evaluated
        return torch.sum(self.get_antibiotics_encodings(self.patient_id, query_time).to(x.device) * self.antibiotic_parameters)
    
    def forward(self, t, x):
        antibiotic_encodings_at_time_t = self.get_antibiotics_encodings(self.patient_id, t).to(x.device)
        x = x.to(torch.float32)
        x = x.reshape(self.input_size)
        x_and_antibiotic_encodings = torch.cat((x, antibiotic_encodings_at_time_t))
        t = t.to(torch.float32)
        t = t.reshape(1)
        x_and_antibiotic_encodings_and_time = torch.cat((x_and_antibiotic_encodings,t))
        return self.net(x_and_antibiotic_encodings_and_time)


class GLVmodel(nn.Module):
    def __init__(self, df_antibiotics, antibiotic_types):
        ''' Parameters:
        df_antibiotics is the antibiotic dataframe
        antibiotic_types is a list of strings giving the names of each antibiotic
        '''
        super(GLVmodel, self).__init__()

        # define attribute that stores the antibiotic information
        # we will store it as a dictionary - lookup table
        # df_antibiotics should be the dataframe containing all of the antibiotic information
        self.antibiotic_types = antibiotic_types
        self.antibiotics_dict = self.preprocess_antibiotics(df_antibiotics, antibiotic_types)
        self.df_antibiotics = df_antibiotics
        
        # current patient_id, to be used in a forward pass, and changed outside the model during training
        self.patient_id = None

    def preprocess_antibiotics(self, df,antibiotic_types, time_resolution=0.1):
        categories = antibiotic_types
        antibiotics_dict = {}

        for _, row in df.iterrows():
            patient_id = row['PatientID']
            start_day = row['StartDayRelativeToNearestHCT']
            stop_day = row['StopDayRelativeToNearestHCT']
            category = row['Category']

            # Initialize dictionary for this patient if not already done
            if patient_id not in antibiotics_dict:
                antibiotics_dict[patient_id] = {}

            # Get indices for the category
            category_idx = categories.index(category)

            # Loop through the discretized time range
            times = np.arange(start_day, stop_day + time_resolution, time_resolution)
            for t in times:
                rounded_t = round(t, int(-np.log10(time_resolution)))  # Round to match resolution
                if rounded_t not in antibiotics_dict[patient_id]:
                    antibiotics_dict[patient_id][rounded_t] = [0] * len(categories)
                antibiotics_dict[patient_id][rounded_t][category_idx] = 1

        return antibiotics_dict
    
    def train(self, df_train):
        # add 1 to everything to get rid of zeros
        df_train.iloc[:, :-3] = df_train.iloc[:, :-3]+1
        df_deltas = pd.DataFrame(columns = ['delta_log_'+taxa for taxa in df_train.columns[:-3]
                                   ] + [taxa + '_start' for taxa in df_train.columns[:-3]
                                       ] + [taxa + '_stop' for taxa in df_train.columns[:-3]
                                       ] + self.antibiotic_types +['start_time', 'stop_time', 'PatientID', 'SampleID_start', 'SampleID_stop'])

        for patient_id in df_train['PatientID'].unique():
            df_patient_id = df_train[df_train['PatientID']==patient_id]
            for i in range(len(df_patient_id)-1):

                start_data = df_patient_id.iloc[i, :]
                stop_data = df_patient_id.iloc[i+1, :]
                # collect start and stop metadata
                start_sample_id = start_data['SampleID']
                stop_sample_id = stop_data['SampleID']
                start_time = start_data['DayRelativeToNearestHCT']
                stop_time = stop_data['DayRelativeToNearestHCT']

                start_abundances = np.array(start_data[:-3])
                stop_abundances = np.array(stop_data[:-3])

                delta = np.log((stop_abundances/start_abundances).astype(np.float64))/(stop_time-start_time)


                df_patient_antibiotics = self.df_antibiotics[self.df_antibiotics['PatientID']==patient_id]
                abx_encodings = []
                for abx in self.antibiotic_types:
                    df_patient_abx_type = df_patient_antibiotics[df_patient_antibiotics['Category']==abx]
                    overlap = int(((df_patient_abx_type["StartDayRelativeToNearestHCT"] <= stop_time) & 
                               (df_patient_abx_type["StopDayRelativeToNearestHCT"] > start_time)).any())
                    abx_encodings.append(overlap)

                row = list(delta) + list(start_abundances
                                  ) + list(stop_abundances
                                          ) + abx_encodings + [start_time] + [stop_time] + [patient_id] + [start_sample_id] + [stop_sample_id]
                df_deltas.loc[len(df_deltas)] = row
        df_deltas = df_deltas.dropna()
        X = pd.concat([df_deltas.iloc[:, 13:26], df_deltas.iloc[:, -20:-5]], axis = 1)
        Y = df_deltas.iloc[:, :13]
        model = make_pipeline(StandardScaler(with_mean=False), RidgeCV())
        model.fit(X, Y)
        ridge_model = model.named_steps['ridgecv']
        scaler = model.named_steps['standardscaler']
        scaled_coefficients = ridge_model.coef_
        original_scale_coefficients = scaled_coefficients / scaler.scale_
        intercept = ridge_model.intercept_
        intercepts, abundance_parameters, antibiotic_parameters = intercept, original_scale_coefficients[:,:-15],original_scale_coefficients[:,-15:]
        self.antibiotic_parameters = torch.from_numpy(antibiotic_parameters).to(device)
        self.abundance_parameters = torch.from_numpy(abundance_parameters).to(device)
        self.intercepts = torch.from_numpy(intercepts).to(device)
        
    def get_antibiotics_encodings(self, patient_id, time, time_resolution=0.1):
        # time comes in as a tensor so we need to convert it to numpy
        time = time.item()
        # note: you need to use resolution 0.1 
        rounded_time = round(time, int(-np.log10(time_resolution)))  # Round to match resolution
        if patient_id in self.antibiotics_dict and rounded_time in self.antibiotics_dict[patient_id]:
            return torch.tensor(self.antibiotics_dict[patient_id][rounded_time], dtype=torch.float32)
        else:
            return torch.zeros(len(self.antibiotic_types), dtype=torch.float32)
        
    def antibiotic_function(self, x, query_time):
        # x is the hidden state of the model, ie the diversity, and is a tensor of shape (1)
        # query_time is the time that this function is being evaluated
        return torch.sum(self.get_antibiotics_encodings(self.patient_id, query_time).to(x.device) * self.antibiotic_parameters).to(device)
    
    def abundances_function(self, x):
        return self.intercepts + torch.matmul(self.abundance_parameters, x)
    
    def forward(self, t, x):
        return torch.matmul(torch.diag(x),self.antibiotic_function(x, t)+self.abundances_function(x)).to(device)
        
     
    

def get_data(df_train, patient_id, simulation_size = False, step_size = 1, sample_id = False, number_of_taxa = 'reduced'):
    '''
    Parameters: 
    df_train is the dataframe we will get data from 
    patient_id is the PatientID we will get data from
    simulation_size is the number of days forward we will get data for
    step_size determines how finely we will simulate the trajectory
    sample_id determines whether we will filter to start the simulation at a particular sample in the trajectory
    '''
    # filter df_train to just that patient
    patient_id_dataframe = df_train[df_train['PatientID'] == patient_id]
    if sample_id:
        patient_id_dataframe = patient_id_dataframe.loc[patient_id_dataframe.index[patient_id_dataframe['SampleID'] == sample_id][0] + 0:]
    # get the diversity trajectory and convert to tensor
    if number_of_taxa == 'reduced':
        y_true = torch.from_numpy(patient_id_dataframe.loc[:, 'Actinobacteria_relative':'<removed_taxa>'].values.astype(float))
    elif number_of_taxa == 'full':
        y_true = torch.from_numpy(patient_id_dataframe.loc[:, 'Actinobacteria_relative':'Verrucomicrobiae_relative'].values.astype(float))
    # get the timepoints of y_true
    t_true = torch.from_numpy(patient_id_dataframe['DayRelativeToNearestHCT'].to_numpy())

    if simulation_size:
        t_start = t_true[0].squeeze()
        # maximum end time
        end_time_bound = t_start + simulation_size
        # filter to be between the start and end times
        t_true_interval = t_true[t_true<=end_time_bound]
        t_true_interval = t_true_interval[t_true_interval>=t_start]
        t_end = t_true_interval[-1]
        if t_start == t_end:
            # if this happens we need to tell the code to skip this iteration 
            return None, None, None, None
        else:
            # generate timepoints for prediction
            prediction_size = int((t_end-t_start)/step_size + 1)
            t_preds = torch.linspace(t_start, t_end, prediction_size)
            # we need to get the indices of the t_true_interval, the true values that we are predicting on and care about, inside t_true
            # we need these so we can subset y_true to get y_true_interval
            y_true_interval_indices = torch.where(t_true[:, None] == t_true_interval)[0]
            true_interval_indices = torch.where(t_preds[:, None] == t_true_interval)[0]
            y_true_interval = y_true[y_true_interval_indices]
            return t_preds.to(device), t_true_interval.to(device), y_true_interval.to(device), true_interval_indices
        
    elif not simulation_size:
        prediction_size = int((t_true[-1]-t_true[0])/step_size + 1)
        # generate timepoints for prediction
        t_preds = torch.linspace(t_true[0], t_true[-1], prediction_size)
        # get the indices of t_true in t_preds
        true_indices = torch.tensor([torch.argmin(torch.abs(t_preds - t)) for t in t_true])
        return t_preds.to(device), t_true.to(device), y_true.to(device), true_indices

       

def loss(batch_y_preds, batch_y_true, weights = torch.ones(13), input_size = 13):
    weights = weights.to(device)
    # pad and packing because these all have different lengths
    padded_group1 = pad_sequence(batch_y_preds, batch_first=True)  # Shape: (batch_size, max_length)
    padded_group2 = pad_sequence(batch_y_true, batch_first=True)  # Shape: (batch_size, max_length)
    # print(f"padded_group1 shape: {padded_group1.shape}")  # Should be (batch_size, max_length)
    # print(f"padded_group2 shape: {padded_group2.shape}")

    max_seq_len = max(padded_group1.shape[1], padded_group2.shape[1])
    # Pad along the second dimension (seq_len)
    padded_group1 = F.pad(padded_group1, (0, 0, 0, max_seq_len - padded_group1.shape[1], 0, 0))  # (left, right, top, bottom, front, back)
    padded_group2 = F.pad(padded_group2, (0, 0, 0, max_seq_len - padded_group2.shape[1], 0, 0))

    # Create a mask for valid (non-padded) values
    mask = (padded_group1 != 0) & (padded_group2 != 0)
    # Compute absolute differences
    differences = torch.abs(padded_group1 - padded_group2)
    # Apply mask and compute mean absolute difference
    masked_differences = differences * mask
    masked_differences = masked_differences*weights.view(1,1,input_size)
    loss = masked_differences.sum() / mask.sum()
    return loss


def wrangle_data(calculate_absolute_abundances = False):
    '''
    This is just preparing all the data for the training
    '''
    df_antibiotics = pd.read_csv('./tbldrug_only_with_antibacterials.csv')
    df_antibiotics.drop('Unnamed: 0', inplace=True, axis = 1)
    antibiotic_types = list(df_antibiotics['Category'].unique())
    df = pd.read_csv('./wrangled_samples_with_negligible_taxa_removed.csv')
    df.drop('Unnamed: 0', inplace=True, axis = 1)
    df_relative_abundances_with_negligible_taxa_removed = pd.concat([df.loc[:, 'Actinobacteria_relative':'Verrucomicrobiae_relative'],
                                    df.loc[:, '<removed_taxa>'],
                                    df.loc[:, 'SampleID':'PatientID'],
                                    df.loc[:, 'DayRelativeToNearestHCT']], axis = 1)
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.sort_values(by=['PatientID', 'DayRelativeToNearestHCT'], ascending=[True, True])
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.dropna(subset=['DayRelativeToNearestHCT'])
    # remove patients with only one sample
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed[df_relative_abundances_with_negligible_taxa_removed.groupby('PatientID')['PatientID'].transform('size')>1]
    
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.drop_duplicates(subset = ['PatientID', 'DayRelativeToNearestHCT'])
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.reset_index(drop= True)
    return df_antibiotics, antibiotic_types, df_relative_abundances_with_negligible_taxa_removed


def wrangle_data_with_qpcr_measurements():
    df = pd.read_csv('./wrangled_samples.csv')
    df.drop('Unnamed: 0', inplace=True, axis = 1)
    df_antibiotics = pd.read_csv('./tbldrug_only_with_antibacterials.csv')
    df_antibiotics.drop('Unnamed: 0', inplace=True, axis = 1)
    antibiotic_types = list(df_antibiotics['Category'].unique())
    df_relative_abundances = pd.concat([df.loc[:, 'Acidobacteriia_relative':'Verrucomicrobiae_relative'],
                                    df.loc[:, 'SampleID':'PatientID'],
                                    df.loc[:, 'DayRelativeToNearestHCT']], axis = 1)
    df_relative_abundances = df_relative_abundances.sort_values(by=['PatientID', 'DayRelativeToNearestHCT'], ascending=[True, True])
    df_relative_abundances = df_relative_abundances.dropna(subset=['DayRelativeToNearestHCT'])
    # remove patients with only one sample
    df_relative_abundances = df_relative_abundances[df_relative_abundances.groupby('PatientID')['PatientID'].transform('size')>1]
    df_relative_abundances = df_relative_abundances.reset_index(drop= True)

    df = pd.read_csv('./wrangled_samples_with_negligible_taxa_removed.csv')
    df.drop('Unnamed: 0', inplace=True, axis = 1)
    df_relative_abundances_with_negligible_taxa_removed = pd.concat([df.loc[:, 'Actinobacteria_relative':'Verrucomicrobiae_relative'],
                                    df.loc[:, '<removed_taxa>'],
                                    df.loc[:, 'SampleID':'PatientID'],
                                    df.loc[:, 'DayRelativeToNearestHCT']], axis = 1)
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.sort_values(by=['PatientID', 'DayRelativeToNearestHCT'], ascending=[True, True])
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.dropna(subset=['DayRelativeToNearestHCT'])
    # remove patients with only one sample
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed[df_relative_abundances_with_negligible_taxa_removed.groupby('PatientID')['PatientID'].transform('size')>1]
    df_relative_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.reset_index(drop= True)

    df_qpcr = pd.read_csv('./tblqpcr.csv')

    df_absolute_abundances_with_negligible_taxa_removed = df_relative_abundances_with_negligible_taxa_removed.merge(df_qpcr, on = 'SampleID', how = 'inner')
    df_absolute_abundances_with_negligible_taxa_removed.loc[:, 'Actinobacteria_relative':'<removed_taxa>'] = df_absolute_abundances_with_negligible_taxa_removed.loc[:, 'Actinobacteria_relative':'<removed_taxa>'].mul(df_absolute_abundances_with_negligible_taxa_removed['qPCR16S'], axis = 0)
    df_absolute_abundances_with_negligible_taxa_removed = df_absolute_abundances_with_negligible_taxa_removed.drop('qPCR16S', axis = 1)

    df_relative_abundances_with_absolute_abundance_measurements_available = df_relative_abundances_with_negligible_taxa_removed[df_relative_abundances_with_negligible_taxa_removed['SampleID'].isin(list(df_absolute_abundances_with_negligible_taxa_removed['SampleID']))]
    df_relative_abundances_with_absolute_abundance_measurements_available

    df_absolute_abundances_with_negligible_taxa_removed = df_absolute_abundances_with_negligible_taxa_removed.drop_duplicates(subset = ['PatientID', 'DayRelativeToNearestHCT']).reset_index(drop=True)
    df_relative_abundances_with_absolute_abundance_measurements_available = df_relative_abundances_with_absolute_abundance_measurements_available.drop_duplicates(subset = ['PatientID', 'DayRelativeToNearestHCT']).reset_index(drop= True)
    return df_absolute_abundances_with_negligible_taxa_removed, df_relative_abundances_with_absolute_abundance_measurements_available



def make_batches(patient_ids, batch_size = 20):
    # returns a list of lists
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)  # Shuffle the list randomly
    return [patient_ids[i:i + batch_size] for i in range(0, len(patient_ids), batch_size)]

def get_batch_prediction_data(model, train_df, batch_patient_ids, method = 'euler', simulation_size =False, sample_wise = False, number_of_taxa = 'reduced'):
    batch_y_preds = []
    batch_y_true = []
    for ID in batch_patient_ids:
        # get the data for this patient trajectory
        if not sample_wise:
            # in this case ID is a patientID
            patient_id = ID
            t_preds, t_true, y_true, true_indices = get_data(train_df, patient_id, simulation_size = simulation_size, number_of_taxa = number_of_taxa)
        elif sample_wise:
            # in this case ID is a sampleID
            patient_id = train_df[train_df['SampleID']==ID]['PatientID'].values[0]
            sample_id = ID
            t_preds, t_true, y_true, true_indices = get_data(train_df, patient_id, simulation_size = simulation_size, sample_id=sample_id, number_of_taxa = number_of_taxa)
        # have to check if the randomly selected interval was empty
        if y_true == None:
            continue
        model.patient_id = patient_id
        y_pred_trajectory = odeint(model, y_true[0], t_preds, method = method)
        y_pred = y_pred_trajectory[true_indices]
        
        batch_y_preds.append(y_pred)
        batch_y_true.append(y_true)
    return batch_y_preds, batch_y_true


def make_domination_predictions(model, test_df, index_of_taxa = 5, number_of_taxa = 'reduced', absolute_abundances = False, prediction_size = 7):
    abundance_predictions = []
    print('computing domination predictions...')
    
    # case that the model is working with relative abundances
    if not absolute_abundances:
        for patient_id in test_df['PatientID'].unique():
            model.patient_id = patient_id
            # get the trajectory data for patient_id's timeline
            t_preds, t_true, y_trues, true_indices = get_data(test_df, patient_id, number_of_taxa = number_of_taxa)
            for i in range(len(t_true)-1):
                # start of predicted trajectory
                t_start = t_true[i]
                y0 = y_trues[i]
                # if there is a domination at time zero, skip
                if y0[index_of_taxa]>=0.3:
                    continue
                t_end = t_start + prediction_size
                number_of_observations_in_the_next_prediction_size_days = ((t_start<t_true)*(t_true<=t_end)).sum()
                y_trues_in_the_next_prediction_size_days = y_trues[i+1:i+number_of_observations_in_the_next_prediction_size_days+1,:]
                # skip this i if there are no observations in the next prediction_size days from t_start
                if y_trues_in_the_next_prediction_size_days.numel()==0:
                    continue
                t_preds = torch.linspace(t_start, t_end, int((t_end-t_start)*3)).to(device)
                y_pred_trajectory = odeint(model, y0, t_preds, method = 'euler')
                taxa_pred_trajectory = y_pred_trajectory[:, index_of_taxa]
                taxa_actual_trajectory = y_trues_in_the_next_prediction_size_days[:, index_of_taxa]

                taxa_pred_domination = np.float64(torch.max(taxa_pred_trajectory).item())
                taxa_actual_domination = np.float64(torch.max(taxa_actual_trajectory).item())

                abundance_predictions.append([taxa_pred_domination, taxa_actual_domination])

        abundance_predictions = np.array(abundance_predictions)
        abundance_predictions[:,1] = abundance_predictions[:,1]>=0.3

    # case that the model is working with absolute abundances
    elif absolute_abundances:
        for patient_id in test_df['PatientID'].unique():
            model.patient_id = patient_id
            # get the trajectory data for patient_id's timeline
            t_preds, t_true, y_trues, true_indices = get_data(test_df, patient_id, number_of_taxa = number_of_taxa)
            for i in range(len(t_true)-1):
                # start of predicted trajectory
                t_start = t_true[i]
                y0 = y_trues[i]
                # Compute the sum of the positive entries of y0
                positive_sum = y0[y0 > 0].sum()
                # make sure the sum is not zero
                if positive_sum != 0:
                    # normalize y0
                    y0_normalized = y0 / positive_sum
                else:
                    continue
                # if there is a domination at time zero, skip
                if y0_normalized[index_of_taxa]>=0.3:
                    continue
                t_end = t_start + prediction_size
                number_of_observations_in_the_next_prediction_size_days = ((t_start<t_true)*(t_true<=t_end)).sum()
                y_trues_in_the_next_prediction_size_days = y_trues[i+1:i+number_of_observations_in_the_next_prediction_size_days+1,:]
                # skip this i if there are no observations in the next prediction_size days from t_start
                if y_trues_in_the_next_prediction_size_days.numel()==0:
                    continue
                t_preds = torch.linspace(t_start, t_end, int((t_end-t_start)*3)).to(device)
                y_pred_trajectory = odeint(model, y0, t_preds, method = 'euler')

                # need to normalize y_pred_trajectory
                # Compute the sum of positive elements for each row (dim=1)
                positive_sums = y_pred_trajectory.where(y_pred_trajectory > 0, 0).sum(dim=1, keepdim=True)

                # Avoid division by zero and perform element-wise division
                y_pred_trajectory_normalized = y_pred_trajectory / positive_sums.where(positive_sums != 0, 1)
    
                # now need to normalze y_trues_in_the_next_prediction_size_days
                positive_sums = y_trues_in_the_next_prediction_size_days.where(y_trues_in_the_next_prediction_size_days > 0, 0).sum(dim=1, keepdim=True)
                y_trues_in_the_next_prediction_size_days_normalized = y_trues_in_the_next_prediction_size_days / positive_sums.where(positive_sums != 0, 1)

                #pick just the taxa of interest
                taxa_pred_trajectory_normalized = y_pred_trajectory_normalized[:, index_of_taxa]
                taxa_actual_trajectory_normalized = y_trues_in_the_next_prediction_size_days_normalized[:, index_of_taxa]

                taxa_pred_domination = np.float64(torch.max(taxa_pred_trajectory_normalized).item())
                taxa_actual_domination = np.float64(torch.max(taxa_actual_trajectory_normalized).item())

                abundance_predictions.append([taxa_pred_domination, taxa_actual_domination])
                
        abundance_predictions = np.array(abundance_predictions)
        abundance_predictions[:,1] = abundance_predictions[:,1]>=0.3

    # print('abundance predictions:', abundance_predictions)
    # Plot ROC curve
    print('plotting ROC curve...')
    y_scores = abundance_predictions[:, 0]  # Predicted probabilities
    y_true = abundance_predictions[:, 1]    # Actual binary labels
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)  # Compute the area under the curve (AUC)
    plt.figure(figsize=(8, 6), num=None)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    taxa_name = test_df.columns[index_of_taxa]
    plt.title(f'ROC Curve for prediction of {taxa_name} domination on held out test data\n in the next {prediction_size} days')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return roc_auc

# def prediction_scatter(x,y,
#                       xlabel = 'True Relative Abundances', ylabel = 'Predicted Relative Abundances',
#                       title = 'NODE relative abundance predictions in the next 7 days on held out test data',
#                       plot = True):
#     if plot:
#         plt.figure(figsize = (10,8))
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.title(title)
#         plt.scatter(x,y, color = 'red', s = 5, alpha = 0.05)
#         slope, intercept, rvalue, pvalue, stddev = stats.linregress(x,y)
#         plt.plot(x, slope*x+intercept, alpha = 0.5, color = 'black')
#         plt.plot(x, x, linestyle = 'dotted', color = 'blue')
#         plt.legend([f'R^2 = {round(rvalue**2,3)}',f'm = {round(slope,3)}'])
#         plt.show()
#         return rvalue**2
#     else:
#         slope, intercept, rvalue, pvalue, stddev = stats.linregress(x,y)
#         return rvalue**2

# def compute_scatter_and_r2(model, test_df, prediction_size = 7, plot=True):
#     test_sample_ids = list(test_df['SampleID'].unique())
#     test_y_preds, test_y_true = get_batch_prediction_data(model, test_df, batch_patient_ids = test_sample_ids , sample_wise = True, simulation_size = prediction_size)


#     predictions_np, trues_np = np.zeros(shape = (13,0)), np.zeros(shape = (13,0))
#     for i in range(len(test_y_preds)):
#         prediction_np = test_y_preds[i].cpu().detach().numpy().transpose()
#         true_np = test_y_true[i].cpu().detach().numpy().transpose()
#         # need to remove the first observation because it is the same in the prediction and the true values
#         prediction_np = prediction_np[:, 1:]
#         true_np = true_np[:, 1:]
#         predictions_np = np.append(predictions_np, prediction_np, axis = 1)
#         trues_np = np.append(trues_np, true_np, axis = 1)
        
#     x = trues_np.flatten()
#     y = predictions_np.flatten()
#     rsquared = prediction_scatter(x,y, plot = plot)

#     return rsquared





# def compute_scatter_and_r2_absolute_abundance_model(model, test_df, prediction_size = 7, plot = True):
#     test_sample_ids = list(test_df['SampleID'].unique())
#     test_y_preds, test_y_true = get_batch_prediction_data(model, test_df, batch_patient_ids = test_sample_ids , sample_wise = True, simulation_size = prediction_size)


#     predictions_np, trues_np = np.zeros(shape = (13,0)), np.zeros(shape = (13,0))
#     for i in range(len(test_y_preds)):
#         prediction_np = test_y_preds[i].cpu().detach().numpy().transpose()
#         true_np = test_y_true[i].cpu().detach().numpy().transpose()
#         # need to remove the first observation because it is the same in the prediction and the true values
#         prediction_np = prediction_np[:, 1:]
#         true_np = true_np[:, 1:]
#         predictions_np = np.append(predictions_np, prediction_np, axis = 1)
#         trues_np = np.append(trues_np, true_np, axis = 1)
        
#     # convert to relative abundances
#     predictions_np[predictions_np<0] = 0
#     predictions_np = predictions_np/predictions_np.sum(axis=0, keepdims = True)
#     trues_np = np.where(trues_np.sum(axis=0, keepdims=True) == 0, 0, trues_np / trues_np.sum(axis=0, keepdims=True))
#     # remove nans
#     valid_rows = (~np.isnan(predictions_np)).any(axis = 0)
#     predictions_np = predictions_np[:, valid_rows]
#     trues_np = trues_np[:, valid_rows]
#     x = trues_np.flatten()
#     y = predictions_np.flatten()
#     rsquared = prediction_scatter(x,y, plot = plot)

#     return rsquared



def prediction_scatter(x, y, classes_array, colors, class_labels,
                      xlabel='True Relative Abundances', 
                      ylabel='Predicted Relative Abundances',
                      title='NODE relative abundance predictions in the next 7 days on held out test data',
                      plot=True, save_eps = False):
    slope, intercept, rvalue, pvalue, stddev = stats.linregress(x, y)
    if plot:
        plt.figure(figsize=(8, 5.6))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = classes_array == i
            if mask.any():
                plt.scatter(x[mask], y[mask], color=color, s=10, alpha=0.1, label=label)
        plt.plot(x, slope*x+intercept, alpha=0.5, color='black')
        plt.plot(x, x, linestyle='dotted', color='blue', label='Perfect prediction')
        
        # Add R² and slope as text annotation
        plt.text(0.05, 0.95, f'R² = {round(rvalue**2, 3)}\nm = {round(slope, 3)}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        for lh in legend.legend_handles[:13]:
            lh.set_alpha(1)
        plt.tight_layout()
        if save_eps:
            plt.savefig(
                f"{title.replace(" ", "_")}.eps",
                format="eps",
                bbox_inches="tight",
                pad_inches=0.1   # adjust padding if you need more/less whitespace
            )
        plt.show()
    return rvalue**2

def compute_scatter_and_r2(model, test_df, prediction_size=7, plot=True, save_eps = False):
    colors = ['#D0D0D0', '#F88379', '#006400', '#16DDD3', '#AD998C', '#D0D0D0', 
              '#AA336A', '#FBA22E', '#D0D0D0', '#EE2C2C', '#D0D0D0', '#CA0BE8', '#808080']
    class_labels = ["Actinobacteria", "Alphaproteobacteria", "Bacilli", "Bacteroidia", "Clostridia",
                   "Coriobacteriia", "Deltaproteobacteria", "Erysipelotrichia", "Fusobacteriia",
                   "Gammaproteobacteria", "Negativicutes", "Verrucomicrobiae", "<removed_taxa>"]
    
    test_sample_ids = list(test_df['SampleID'].unique())
    test_y_preds, test_y_true = get_batch_prediction_data(model, test_df, batch_patient_ids=test_sample_ids, 
                                                                    sample_wise=True, simulation_size=prediction_size)
    predictions_np, trues_np = np.zeros(shape=(13, 0)), np.zeros(shape=(13, 0))
    for i in range(len(test_y_preds)):
        prediction_np = test_y_preds[i].cpu().detach().numpy().transpose()[:, 1:]
        true_np = test_y_true[i].cpu().detach().numpy().transpose()[:, 1:]
        predictions_np = np.append(predictions_np, prediction_np, axis=1)
        trues_np = np.append(trues_np, true_np, axis=1)
    
    classes_array = np.repeat(np.arange(13), predictions_np.shape[1])
    x, y = trues_np.flatten(), predictions_np.flatten()
    return prediction_scatter(x, y, classes_array, colors, class_labels, plot=plot, save_eps = save_eps)

def compute_scatter_and_r2_absolute_abundance_model(model, test_df, prediction_size=7, plot=True, save_eps = False):
    colors = ['#D0D0D0', '#F88379', '#006400', '#16DDD3', '#AD998C', '#D0D0D0', 
              '#AA336A', '#FBA22E', '#D0D0D0', '#EE2C2C', '#D0D0D0', '#CA0BE8', '#808080']
    class_labels = ["Actinobacteria", "Alphaproteobacteria", "Bacilli", "Bacteroidia", "Clostridia",
                   "Coriobacteriia", "Deltaproteobacteria", "Erysipelotrichia", "Fusobacteriia",
                   "Gammaproteobacteria", "Negativicutes", "Verrucomicrobiae", "<removed_taxa>"]
    test_sample_ids = list(test_df['SampleID'].unique())
    test_y_preds, test_y_true = get_batch_prediction_data(model, test_df, batch_patient_ids=test_sample_ids, 
                                                                    sample_wise=True, simulation_size=prediction_size)
    predictions_np, trues_np = np.zeros(shape=(13, 0)), np.zeros(shape=(13, 0))
    for i in range(len(test_y_preds)):
        prediction_np = test_y_preds[i].cpu().detach().numpy().transpose()[:, 1:]
        true_np = test_y_true[i].cpu().detach().numpy().transpose()[:, 1:]
        predictions_np = np.append(predictions_np, prediction_np, axis=1)
        trues_np = np.append(trues_np, true_np, axis=1)
    
    # Convert to relative abundances
    predictions_np[predictions_np < 0] = 0
    pred_sums = predictions_np.sum(axis=0, keepdims=True)
    pred_sums[pred_sums == 0] = 1  # Avoid division by zero
    predictions_np = predictions_np / pred_sums
    true_sums = trues_np.sum(axis=0, keepdims=True)
    true_sums[true_sums == 0] = 1  # Avoid division by zero
    trues_np = trues_np / true_sums
    
    # Remove columns with NaNs
    valid_cols = ~(np.isnan(predictions_np).any(axis=0) | np.isnan(trues_np).any(axis=0))
    predictions_np = predictions_np[:, valid_cols]
    trues_np = trues_np[:, valid_cols]
    
    classes_array = np.repeat(np.arange(13), predictions_np.shape[1])
    x, y = trues_np.flatten(), predictions_np.flatten()
    return prediction_scatter(x, y, classes_array, colors, class_labels, plot=plot, title = 'GLV relative abundance predictions in the next 7 days on held out test data', save_eps = save_eps)



def compute_baseline_scatter_and_r2(test_df, prediction_size=7, plot=True, save_eps = False):
    colors = ['#D0D0D0', '#F88379', '#006400', '#16DDD3', '#AD998C', '#D0D0D0', 
              '#AA336A', '#FBA22E', '#D0D0D0', '#EE2C2C', '#D0D0D0', '#CA0BE8', '#808080']
    class_labels = ["Actinobacteria", "Alphaproteobacteria", "Bacilli", "Bacteroidia", "Clostridia",
                   "Coriobacteriia", "Deltaproteobacteria", "Erysipelotrichia", "Fusobacteriia",
                   "Gammaproteobacteria", "Negativicutes", "Verrucomicrobiae", "<removed_taxa>"]
    
    abundance_cols = [col for col in test_df.columns if col not in ['SampleID', 'PatientID', 'DayRelativeToNearestHCT']][:13]
    
    test_sample_ids = list(test_df['SampleID'].unique())
    predictions_np, trues_np = np.zeros(shape=(13, 0)), np.zeros(shape=(13, 0))
    
    for sample_id in test_sample_ids:
        patient_id = test_df[test_df['SampleID']==sample_id]['PatientID'].values[0]
        patient_data = test_df[test_df['PatientID']==patient_id].sort_values('DayRelativeToNearestHCT')
        sample_row = patient_data[patient_data['SampleID']==sample_id]
        
        initial_abundance = sample_row[abundance_cols].values[0]
        initial_day = sample_row['DayRelativeToNearestHCT'].values[0]
        
        # Get future observations within prediction_size days
        future_data = patient_data[(patient_data['DayRelativeToNearestHCT'] > initial_day) & 
                                   (patient_data['DayRelativeToNearestHCT'] <= initial_day + prediction_size)]
        if len(future_data) == 0:
            continue
            
        future_abundances = future_data[abundance_cols].values.T
        baseline_predictions = np.tile(initial_abundance.reshape(-1, 1), (1, future_abundances.shape[1]))
        
        predictions_np = np.append(predictions_np, baseline_predictions, axis=1)
        trues_np = np.append(trues_np, future_abundances, axis=1)
    
    classes_array = np.repeat(np.arange(13), predictions_np.shape[1])
    x, y = trues_np.flatten(), predictions_np.flatten()
    return prediction_scatter(x, y, classes_array, colors, class_labels, plot=plot, title = 'Initial abundance predictions vs. true in the next 7 days', save_eps = save_eps)
