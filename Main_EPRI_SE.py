
### Objective - State Estimation from a limited number of PMU measurements using Machine Learning


##################################################################################################
## Importing neccessary libraries
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import math
##################################################################################################

##################################################################################################
### Loading the functions used in this main file
from Data_Loading import DeNSE_data_loader
from Data_pre_processing import PMU_current_flow_identifier  ## Based on PMU locations, determines the line flows that are directly observable
from Data_pre_processing import Generate_noisy_measurements_Gaussian_noise ## Generate Gaussian noisy measurements by adding Gaussian noise
from Data_pre_processing import Input_Data_Normalization ## Normalizing the input data
from Data_pre_processing import Data_Removing_NaNs  ## Removing NaNs, if any, from the input and output data
from DNN_training_for_state_estimation import DNN_training_base_topology  ## Training the DNN based on the hyperparameters defined
from Post_processing_tasks import Post_processing
#%%
## Seeding to obtain replicable results
seed_value= 250
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

#%%

## Loading the required data
[(df_If_mag, df_If_ang, df_It_mag,  df_It_ang, df_VDATA_mag, df_VDATA_ang] =  DeNSE_data_loader()

  
##################################################################################################
## Data Pre-processing stage (using training data)
## Buses where PMUs are placed
pmu_loc1 = [8,9,10,26,30,38,63,64,65,68,81]
pmu_loc1_python_index = (np.asarray(pmu_loc1)-1).tolist()

## Extracting From and To branch information for PMU placed buses
[From_branches_req, To_branches_req] = PMU_current_flow_identifier(pmu_loc1, df_From_To_buses_118)

#%% Gaussian Noise values (Standard Deviation)
sigma_mag = 0.01*2/6
sigma_ang = 0.5*2/6


# Adding Noise to training data
[df_Input_NN, df_Output_NN] =  Generate_noisy_measurements_Gaussian_noise(sigma_mag, sigma_ang, df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118, pmu_loc1,pmu_loc1_python_index, From_branches_req, To_branches_req)
df_Input_NN_train = df_Input_NN
# Normalizing and Removing NANs 
df_Input_NN_normalized = Input_Data_Normalization(df_Input_NN, df_Input_NN_train)
[df_Input_NN_normalized, df_Output_NN] =  Data_Removing_NaNs(df_Input_NN_normalized, df_Output_NN)
x = df_Input_NN_normalized.values # x- input matrix
y = df_Output_NN.values # y - output matrix
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state= 8) #
##################################################################################################
#%% Training Deep Neural Network Model
model = DNN_training_base_topology(X_train, X_val, y_train, y_val, x, y)
#%% Data Preprocesing Stage-Testing Data
df_If_mag_test = pd.read_csv(filepath+('#Testing_From_Bus_Magnitude_Data'), header = None)
df_If_ang_test = pd.read_csv('#Testing_From_Bus_PhaseAngle_Data'), header = None)
df_It_mag_test= pd.read_csv('#Testing_To_Bus_Magnitude_Data'), header = None)
df_It_ang_test = pd.read_csv('#Testing_To_Bus_PhaseAngle_Data'), header = None)
df_VDATA_mag_test = pd.read_csv('#Testing_Voltage_Magnitude_Data'), header = None)
df_VDATA_ang_test = pd.read_csv('#Testing_Voltage_PhaseAngle_Data'), header = None)
# Extracting PMU placed buses data
df_VDATA_ang_renamed_whole = df_VDATA_ang_test.rename(columns={x:y for x,y in zip(df_VDATA_ang_test.columns,range(0+118,len(df_VDATA_ang_test.columns)+118))}) 
df_V_mag_PMU_test = df_VDATA_mag_test[df_VDATA_mag_test.columns[pmu_loc1_python_index]]
df_V_ang_PMU_test = df_VDATA_ang_test[df_VDATA_ang_test.columns[pmu_loc1_python_index]]

df_If_mag_PMU_test = df_If_mag_test[df_If_mag_test.columns[From_branches_req]]
df_If_ang_PMU_test = df_If_ang_test[df_If_ang_test.columns[From_branches_req]]
df_It_mag_PMU_test = df_It_mag_test[df_It_mag_test.columns[To_branches_req]]
df_It_ang_PMU_test = df_It_ang_test[df_It_ang_test.columns[To_branches_req]]
df_Input_NN_test = pd.concat([df_V_mag_PMU_test, np.deg2rad(df_V_ang_PMU_test), df_If_mag_PMU_test, np.deg2rad(df_If_ang_PMU_test), df_It_mag_PMU_test, np.deg2rad(df_It_ang_PMU_test) ], axis=1) # Concatenating voltage magnitude and angles to get input

df_Output_NN_test = pd.concat([df_VDATA_mag_test,np.deg2rad(df_VDATA_ang_renamed_whole)], axis=1)

# Normalizing and Removing NANs
df_Input_NN_normalized_test = Input_Data_Normalization(df_Input_NN_test, df_Input_NN_train)
[df_Input_NN_normalized_test, df_Output_NN_test] =  Data_Removing_NaNs(df_Input_NN_normalized_test, df_Output_NN_test)
X_test = df_Input_NN_normalized_test.values # x- input matrix
y_test = df_Output_NN_test.values # y - output 


#%% Predicted State Estimates
pred = model.predict(X_test)
#%% Visualizing DeNSE performance
pmu_loc_np = np.asarray(pmu_loc1) # pmu_loc_np - numpy version of pmu locaiton indices

    
Entire_buses = np.zeros((118,1)) # Initializing the Lcoation of all buses 
    
for q in range(118):
    Entire_buses[q] = q+1 # Lcoation of all buses (numpy array from 1 to 118)
      
      


[Mag_MAE, Angle_MAE] = Post_processing(y_test, pred)
