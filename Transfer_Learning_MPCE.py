# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 08:58:29 2023

@author: hshah59
"""
#%%Add this to the Funcion file#

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import load_model
import time  




def DNN_training_TransferLearning(X_train, X_val, y_train, y_val, x, y):
        
    # Build the neural network
    model = Sequential()
    model.add(Dense(500, input_dim=x.shape[1], activation='relu')) # Hidden 1
    model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 2
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 3
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 4
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(y.shape[1])) # Output
    model.load_weights("D:\Hritik/newest_692023_51.h5")
    model.compile(loss='mean_squared_error',metrics = ['mae'], optimizer='adam')
    #monitor = EarlyStopping(monitor='val_mae',mode ='min', min_delta=0.0000001, patience=30, verbose=1, baseline=0.1070)
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=30, mode='min', min_delta=0.0000001)
    start = time.time()
    history = model.fit(X_train,y_train,verbose=1,epochs=90,validation_data = (X_val,y_val),batch_size=512, callbacks=[reduce_lr])
    end = time.time()
    elapsed_time = end-start
    print(elapsed_time)
    #model.save_weights("D:\Hritik\Saved_models\Base_TF_weights_TF.h5")
    #model.save("D:\Hritik\Saved_models\Base_topology_DNN_TF.h5")
    return(model)

#%% INput data for all topologies
filepath = r'D:\Hritik and Antos_Data Generation for 118-bus\DeNSE_10k_with_TCTR_corrected\Datagen_10k_TCTR\DeNSE_10k_Train_Likely_Topologies_TCTR/'


# i =input("Enter the dataset: ")
# # Load_Training_Data = "Load_Training_Data_" + i + "(filepath)"
# # Load_Training_Data
# if(i == "T1"):
#     T = Load_Training_Data_T1(filepath)
# elif(i=="T2"):
#     T = Load_Training_Data_T2(filepath)
# elif(i=="T3"):
#     T = Load_Training_Data_T3(filepath)
# elif(i=="T4"):
#     T = Load_Training_Data_T4(filepath)

i =input("Enter the dataset: ")

T = Load_Training_Data(filepath, i)


[df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118] = T

#%% Data Preprocessing [ Some functions defined below are already there in files you have]
pmu_loc1 = [8,9,10,26,30,38,63,64,65,68,81]


pmu_loc1_python_index = (np.asarray(pmu_loc1)-1).tolist()

[From_branches_req, To_branches_req] = PMU_current_flow_identifier(pmu_loc1, df_From_To_buses_118)
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
model = DNN_training_TransferLearning(X_train, X_val, y_train, y_val, x, y)
