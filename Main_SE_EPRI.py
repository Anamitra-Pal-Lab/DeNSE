### Code by Antos Varghese
### Objective - State Estimation from a limited number of PMU measurements using 


##################################################################################################
## Importing neccessary libraries
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
##################################################################################################

##################################################################################################
### Loading the functions used in this main file
from Load_Data import Load_Training_Data ## Loads the training data (voltages and currents used)
from Data_Pre_Processing import PMU_current_flow_identifier ## Based on PMU locations, determines the line flows that are directly observable
from Data_Pre_Processing import Generate_noisy_measurements_Gaussian_noise ## Generate Gaussian noisy measurements by adding Gaussian noise
from Data_Pre_Processing import Input_Data_Normalization1 ## Normalizing the input data
from Data_Pre_Processing import Data_Removing_NaNs ## Removing NaNs, if any, from the input and output data
from DNN_training_SE import  DNN_training ## Training the DNN based on the hyperparameters defined
from Load_Data import Load_Testing_Data ## Loads the testing data (voltages and currents used)
from Data_Post_Processing import Estimation_Error_MAE_viz
##################################################################################################


#%%
##################################################################################################
## Seeding to obtain replicable results
seed_value= 250
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
##################################################################################################




##################################################################################################
## Data Pre-processing stage (using training data)
filepath = r'D:/Antos/Dropbox (ASU)/EPRI_Project/New_PMU_data_0325/Train/' ### File path to the training data
[df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118] = Load_Training_Data(filepath) 
pmu_loc1 = [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] 
[From_branches_req, To_branches_req] = PMU_current_flow_identifier(pmu_loc1, df_From_To_buses_118)

sigma_mag = 0.01*2/6 # Noise in Magnitude standard deviation
sigma_ang = 0.5*2/6
[df_Input_NN, df_Output_NN] =  Generate_noisy_measurements_Gaussian_noise(sigma_mag, sigma_ang, df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118, pmu_loc1, From_branches_req, To_branches_req)

df_Input_NN_normalized = Input_Data_Normalization1(df_Input_NN)
[df_Input_NN_normalized, df_Output_NN] =  Data_Removing_NaNs(df_Input_NN_normalized, df_Output_NN)
X = df_Input_NN_normalized.values # x- input matrix
y = df_Output_NN.values # y - output matrix
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state= 8) #
##################################################################################################


##################################################################################################
### DNN training section
### DNN Hyper parameters
n_epochs = 100  ## Number of epochs
n_neurons_per_layer = 800 ## Number of neurons per layer
n_hidden_layers = 5 ## Number of hidden layers

model =  DNN_training(X_train, X_val, y_train, y_val, n_epochs, n_neurons_per_layer, n_hidden_layers)
##################################################################################################

#%%

filepath = r'D:/Antos/Dropbox (ASU)/EPRI_Project/New_PMU_data_0325/Test/'
[df_VDATA_mag_test, df_VDATA_ang_test, df_If_mag_test, df_If_ang_test, df_It_mag_test, df_It_ang_test] = Load_Testing_Data(filepath)

[df_Input_NN_test, df_Output_NN_test] =  Generate_noisy_measurements_Gaussian_noise(sigma_mag, sigma_ang, df_VDATA_mag_test, df_VDATA_ang_test, df_If_mag_test, df_If_ang_test, df_It_mag_test, df_It_ang_test, df_From_To_buses_118, pmu_loc1, From_branches_req, To_branches_req)
df_Input_NN_normalized_test = Input_Data_Normalization1(df_Input_NN_test)
[df_Input_NN_normalized_test, df_Output_NN_test] =  Data_Removing_NaNs(df_Input_NN_normalized_test, df_Output_NN_test)
X_test = df_Input_NN_normalized_test.values # x- input matrix
y_test = df_Output_NN_test.values # y - output matrix


pred = model.predict(X_test)

Estimation_Error_MAE_viz(X_test, y_test, pred, pmu_loc1)

