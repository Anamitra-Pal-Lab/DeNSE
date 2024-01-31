
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
from Data_Loading import DeNSE_data_loader_training_data
from Data_Loading import DeNSE_data_loader_testing_data
from Data_pre_processing import PMU_current_flow_identifier  ## Based on PMU locations, determines the line flows that are directly observable
from Data_pre_processing import Generate_noisy_measurements_Gaussian_noise ## Generate Gaussian noisy measurements by adding Gaussian noise
from Data_pre_processing import Input_Data_Normalization ## Normalizing the input data
from Data_pre_processing import Data_Removing_NaNs  ## Removing NaNs, if any, from the input and output data
from DNN_training_for_state_estimation import DNN_training_base_topology  ## Training the DNN based on the hyperparameters defined
from DNN_training_for_state_estimation import DNN_training_TransferLearning
from Post_processing_tasks import Post_processing
from Post_processing_tasks import Bad_Data_injection_sample_wise_DeNSE
from Post_processing_tasks import The_Wald_Test
from Post_processing_tasks import Novel_BDR_w_NOC_samplewise_DeNSE
from Post_processing_tasks import Bad_Data_replacement_with_mean_samplewise_DeNSE
from Post_processing_tasks import Estimation_Error_MAE_viz

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
[df_If_mag, df_If_ang, df_It_mag,  df_It_ang, df_VDATA_mag, df_VDATA_ang] =  DeNSE_data_loader_training_data()

  #%%
##################################################################################################
## Data Pre-processing stage (using training data)
## Buses where PMUs are placed
pmu_loc1 = [8,9,10,26,30,38,63,64,65,68,81]
pmu_loc1_python_index = (np.asarray(pmu_loc1)-1).tolist()

df_From_To_buses_118 = pd.read_csv('From_To_buses_118.csv', header = None) # The from bus- to bus information

#pmu_loc1 =  pd.read_csv('PMU_locations.csv', header = None) # the PMU locations as input


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

[df_If_mag_test, df_If_ang_test, df_It_mag_test,  df_It_ang_test, df_VDATA_mag_test, df_VDATA_ang_test] =  DeNSE_data_loader_testing_data()


#%%
# Extracting PMU placed buses data
df_VDATA_ang_renamed_whole = df_VDATA_ang_test.rename(columns={x:y for x,y in zip(df_VDATA_ang_test.columns,range(0+118,len(df_VDATA_ang_test.columns)+118))}) 
df_V_mag_PMU_test = df_VDATA_mag_test[df_VDATA_mag_test.columns[pmu_loc1_python_index]]
df_V_ang_PMU_test = df_VDATA_ang_test[df_VDATA_ang_test.columns[pmu_loc1_python_index]]

df_If_mag_PMU_test = df_If_mag_test[df_If_mag_test.columns[From_branches_req]]
df_If_ang_PMU_test = df_If_ang_test[df_If_ang_test.columns[From_branches_req]]
df_It_mag_PMU_test = df_It_mag_test[df_It_mag_test.columns[To_branches_req]]
df_It_ang_PMU_test = df_It_ang_test[df_It_ang_test.columns[To_branches_req]]
df_Input_NN_test = pd.concat([df_V_mag_PMU_test, np.deg2rad(df_V_ang_PMU_test), df_If_mag_PMU_test, np.deg2rad(df_If_ang_PMU_test), df_It_mag_PMU_test, np.deg2rad(df_It_ang_PMU_test) ], axis=1) # Concatenating voltage magnitude and angles to get input
#%%
df_Output_NN_test = pd.concat([df_VDATA_mag_test,np.deg2rad(df_VDATA_ang_renamed_whole)], axis=1)


headings = df_Input_NN_train.columns.tolist()
df_Input_NN_test.columns = headings

# Normalizing and Removing NANs
df_Input_NN_normalized_test = Input_Data_Normalization(df_Input_NN_test, df_Input_NN_train)

#%%
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
      
      

[Mag_MAPE, Angle_MAE] = Post_processing(y_test, pred, Entire_buses)

#%%




## Creating Test data with bad data by injecting bad values to some of the features in every sample randomly - X_test_with_BD

Prob_bad_Data = 0.2
Std_Bad_data = 3
Mean_bad_data = 3

X_test_with_BD = np.zeros((X_test.shape[0], X_test.shape[1])) # initializing the variable of test data with bad data

Bad_data_feature_np_actual = np.zeros((X_test.shape[0]*X_test.shape[1], 2)) 
BD_net_counter = 0
for i in range(X_test.shape[0]):
    print(i)
    BD_curr_counter = 0 
    [X_test_sample_with_BD, bad_data_feature_index] = Bad_Data_injection_sample_wise_DeNSE(X_test[i,:], Prob_bad_Data, Std_Bad_data, Mean_bad_data)
    X_test_with_BD[i,:] = X_test_sample_with_BD
    for temp in range(len(bad_data_feature_index)):
        Bad_data_feature_np_actual[BD_net_counter, 0] = i
        Bad_data_feature_np_actual[BD_net_counter, 1] = bad_data_feature_index[BD_curr_counter]
        BD_curr_counter = BD_curr_counter + 1
        BD_net_counter = BD_net_counter + 1

#%%
## comparing the actual bad data indices with those flagged as bad data by Wald test

Bad_data_feature_actual =   Bad_data_feature_np_actual[np.nonzero(Bad_data_feature_np_actual[:,1])]  
Bad_data_index_wald_test = The_Wald_Test(X_test_with_BD)
Bad_data_index_wald_test_np  = np.zeros((len(Bad_data_index_wald_test[0]), 2))
Bad_data_index_wald_test_np[:,0] = Bad_data_index_wald_test[0]
Bad_data_index_wald_test_np[:,1] = Bad_data_index_wald_test[1]


#%% BD replacement with mean
#pmu_loc1 = pmu_loc1[0].values

X_test_bad_replaced_w_mean = np.copy(X_test_with_BD)
for sample_num in range(X_test_with_BD.shape[0]):
    X_test_bad_sample = X_test_with_BD[sample_num, :] 
    bad_data_index_wald_test_sample  = The_Wald_Test(X_test_bad_sample) 
    X_test_bad_sample_replaced_w_mean =Bad_Data_replacement_with_mean_samplewise_DeNSE(X_test_bad_sample, bad_data_index_wald_test_sample)
    X_test_bad_replaced_w_mean[sample_num, :] =  X_test_bad_sample_replaced_w_mean

load = load_model("Base_topology_DNN_TF_11.h5")

pred_bad_replaced_w_mean = load.predict(X_test_bad_replaced_w_mean)
#MSE_BD_replaced_with_mean = mean_squared_error(y_test, pred_bad_replaced_w_mean)
#[Mag_MAPE_rwm, Angle_MAE_rwm] = Estimation_Error_MAE_viz(X_test_bad_replaced_w_mean, y_test, pred_bad_replaced_w_mean , pmu_loc1) # the estimation errors when bad data is replaced with mean

[Mag_MAPE_rwm, Angle_MAE_rwm] = Estimation_Error_MAE_viz(X_test_bad_replaced_w_mean,   y_test, pred_bad_replaced_w_mean , pmu_loc1)
#%% BD replacement with Nearesr Operating Condition

exec_time = []
X_test_bad_replaced_with_nearest_OC = np.copy(X_test_with_BD)
for sample_num in range(X_test_with_BD.shape[0]):
    print(str(sample_num*100/X_test_with_BD.shape[0]) + "% BDR with NOC completed")
    t = time.time()
    X_test_bad_sample = X_test_with_BD[sample_num, :] 
    bad_data_index_wald_test_noc  = The_Wald_Test(X_test_bad_sample)  
    ibfs = np.asarray(bad_data_index_wald_test_noc).T
    X_test_bad_sample_rw_NOC = Novel_BDR_w_NOC_samplewise_DeNSE(X_test_bad_sample, X_train, ibfs)
    X_test_bad_replaced_with_nearest_OC[sample_num, :] =  X_test_bad_sample_rw_NOC
    elapsed = time.time() - t
    print(elapsed)
    exec_time.append(elapsed)
load = load_model("Base_topology_DNN_TF_11.h5")
pred_bad_replaced_w_NOC = load.predict(X_test_bad_replaced_with_nearest_OC)
#%%
[Mag_MAPE_rwNOC, Angle_MAE_rwNOC] = Estimation_Error_MAE_viz(X_test_bad_replaced_with_nearest_OC,   y_test, pred_bad_replaced_w_NOC , pmu_loc1)

#%% Transfer Learning Section (TL)


i =input("Enter the dataset: ")
filepath = r"Transfer_Learning_Data_Training" # path to Transfer Learning data folder from dropbox weblink
T = Load_Training_Data(filepath, i)


[df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang] = T



[From_branches_req, To_branches_req] = PMU_current_flow_identifier(pmu_loc1, df_From_To_buses_118)
sigma_mag = 0.01*2/6
sigma_ang = 0.5*2/6


[df_Input_NN, df_Output_NN] =  Generate_noisy_measurements_Gaussian_noise(sigma_mag, sigma_ang, df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118, pmu_loc1,pmu_loc1_python_index, From_branches_req, To_branches_req)
df_Input_NN_train = df_Input_NN

df_Input_NN_normalized = Input_Data_Normalization(df_Input_NN, df_Input_NN_train)
[df_Input_NN_normalized, df_Output_NN] =  Data_Removing_NaNs(df_Input_NN_normalized, df_Output_NN)
x = df_Input_NN_normalized.values # x- input matrix
y = df_Output_NN.values # y - output matrix
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state= 8) #


Transfer_learning_model = DNN_training_TransferLearning(X_train, X_val, y_train, y_val, x, y)

filepath_test = r"Transfer_Learning_Data_Testing" # path to Transfer Learning data folder from dropbox weblink
T_t = Load_Testing_Data(filepath_test
                        , i)
[df_VDATA_mag_test, df_VDATA_ang_test, df_If_mag_test, df_If_ang_test, df_It_mag_test, df_It_ang_test] = T_t
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
X_test_TL = df_Input_NN_normalized_test.values # x- input matrix
y_test_TL = df_Output_NN_test.values # y - output 
predicted_TL_state_esimates = load.predict(X_test_TL)
[Mag_MAPE_TL, Angle_MAE_TL] = Estimation_Error_MAE_viz(X_test_TL,   y_test_TL, predicted_TL_state_esimates , pmu_loc1)






