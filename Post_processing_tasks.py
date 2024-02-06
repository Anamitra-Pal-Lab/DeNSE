import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scipy import stats
import numpy as np
import pandas as pd
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scipy import stats
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import load_model
import time  

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import math


def Post_processing(y_test, pred, Entire_buses):
    Mag_MAE =np.mean(abs(y_test[:,0:118] - pred[:,0:118]), axis=0)*100
    print(np.mean(Mag_MAE))
    Angle_MAE = np.mean(abs(y_test[:,118:236] - pred[:,118:236]), axis=0)
    print(np.mean(Angle_MAE))
    pd.DataFrame(Mag_MAE).to_csv('Mag_MAPE_118.csv')
    pd.DataFrame(Angle_MAE).to_csv('Angle_MAE_118.csv')
              
    # barlist1 = plt.bar(Entire_buses[:,0], Mag_MAE , color ='blue',width = 0.8)
    # plt.xlabel('Bus number')
    # plt.ylabel('Voltage Magnitude Error (MAPE)')
    # plt.savefig('Vmag_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    # plt.show()
        
        
    # barlist1 = plt.bar(Entire_buses[:,0], Angle_MAE, color ='blue',width = 0.8)
    # plt.xlabel('Bus number')
    # plt.ylabel('Voltage Angle Error (MAE)')
    # plt.savefig('Vang_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    # plt.show()
    
    return(Mag_MAE, Angle_MAE)



def Bad_Data_injection_sample_wise_DeNSE(X_test_sample, Prob_bad_Data, Std_Bad_data, Mean_bad_data): # to inject bad data samplewise tho test data, to verify how the bad data detection and replacement scheme works
    
    # Prob_bad_Data - is the probability of bad data in the data set
    # Std_Bad_data - standard deviation by which data is multiplied so that the bad data is generated
    # Mean_bad_data - mean of bad data distribution (suggested + 3 or -3)
    counter = 0 # to count how many features have been injected by bad data
    bad_data_index = [] # to store the index of bad data for verification
    
    X_test_sample_with_BD = np.copy(X_test_sample) ### This line initializing the output vector (after the BD is added to some features)
    bad_data_feature_index = []
    for j in range(0, X_test_sample.shape[0]): # iterating over every feature
        if np.random.binomial(1, Prob_bad_Data, 1)==1: # i.e. if current feature is to be added bad data
            base_BD_value = np.random.randn(1, 1)
            if(base_BD_value>=0):
                X_test_sample_with_BD[j] =   Mean_bad_data + (base_BD_value*Std_Bad_data)
            if(base_BD_value<0):
                X_test_sample_with_BD[j] =   (-1*Mean_bad_data) + (base_BD_value*Std_Bad_data)
            bad_data_feature_index.append(j)
            counter = counter+1
    print(counter)
    return(X_test_sample_with_BD, bad_data_feature_index)

def The_Wald_Test(X_test_bad):
    # Wald test - Identify bad data - Replace it with mean

    false_alarm_level_1 = 2.5758  #  1% level percentage
    false_alarm_level_2 = 2.3263  #  2% level percentage
    false_alarm_level_3 = 2.1701  #  3% level percentage
    false_alarm_level_4 = 2.0537  #  4% level percentage
    false_alarm_level_5 = 1.9600  #  5% level percentage

    bad_data_index_wald_test = np.where(abs(X_test_bad)  > false_alarm_level_1)
    
    return(bad_data_index_wald_test)


def Novel_BDR_w_NOC_samplewise_DeNSE(X_test_bad_sample, X_train, ibfs):
    iafs = np.zeros((X_test_bad_sample.shape[0],1)) # Initializing the Lcoation of all buses 
    for q in range(X_test_bad_sample.shape[0]):
      iafs[q] = q # iafs - index of all feature set
    X_test_bad_sample_copy = np.copy(X_test_bad_sample) # creating a copy of test data with bad data
    X_train_copy = np.copy(X_train) # creating a copy of the training input data set
    X_test_bad_sample_replaced_with_nearest_OC = np.copy(X_test_bad_sample_copy)  # initializing the output vector (after reaplcing the bad data with nearest OC values)
    igfs = np.delete(iafs,ibfs) # finding the indics of good features (igfs)
    igfs = igfs.astype(np.int64)
    X_train_L2_score_curr_sample =np.sum(((X_train_copy[:,igfs] - X_test_bad_sample_copy[igfs])**2), axis=1)**(1./2)
    nearest_OC_curr_sample = np.argmin(X_train_L2_score_curr_sample)
    X_test_bad_sample_replaced_with_nearest_OC[ibfs] = X_train_copy[nearest_OC_curr_sample,ibfs]
    return(X_test_bad_sample_replaced_with_nearest_OC)
    
    
def Bad_Data_replacement_with_mean_samplewise_DeNSE(X_test_bad_sample, bad_data_index_wald_test_sample):
    X_test_bad_sample_replaced_w_mean = np.copy(X_test_bad_sample)
    X_test_bad_sample_replaced_w_mean[bad_data_index_wald_test_sample] = 0
    return(X_test_bad_sample_replaced_w_mean)


def Estimation_Error_MAE_viz(X_test, y_test, pred, pmu_loc1): # estimating MAE of amgnitude and angle; also plots of both
    
    pmu_loc_np = np.asarray(pmu_loc1) # pmu_loc_np - numpy version of pmu locaiton indices
    #pmu_loc_np_python = pmu_loc_np - 1
    
    Entire_buses = np.zeros((118,1)) # Initializing the Lcoation of all buses 
    
    for q in range(118):
      Entire_buses[q] = q+1 # Lcoation of all buses (numpy array from 1 to 118)
      
      
    PMU_unobserved_buses = Entire_buses
    mask = np.isin(Entire_buses, pmu_loc_np, invert=True)
    PMU_unobserved_buses = Entire_buses[mask] # PMU_unobserved_buses - locaions PMU unobserved buses
    PMU_unobserved_buses = PMU_unobserved_buses.astype(np.int64)
    
    
    print((y_test[:,pmu_loc_np-1] - pred[:,pmu_loc_np-1]).shape)
    print((y_test[:,PMU_unobserved_buses-1] - pred[:,PMU_unobserved_buses-1]).shape)

    #Calculating the MAE of PMU pbserved and unobserved buses separately for magnitude and phase angle variables
    Mag_MAE_PMU = np.mean(abs(y_test[:,pmu_loc_np-1] - pred[:,pmu_loc_np-1]), axis=0)
    print(np.mean(Mag_MAE_PMU))
    Mag_MAE_non_PMU = np.mean(abs(y_test[:,PMU_unobserved_buses-1] - pred[:,PMU_unobserved_buses-1]), axis=0)
    print(np.mean(Mag_MAE_non_PMU))
    Angle_MAE_PMU = np.mean(abs(y_test[:,pmu_loc_np+118-1] - pred[:,pmu_loc_np+118-1]), axis=0)
    print(np.mean(Angle_MAE_PMU))
    Angle_MAE_non_PMU = np.mean(abs(y_test[:,PMU_unobserved_buses+118-1] - pred[:,PMU_unobserved_buses+118-1]), axis=0)
    print(np.mean(Angle_MAE_non_PMU))
    
    Angle_MAE = np.mean(abs(y_test[:,0:118] - pred[:,0:118]), axis=0)
    print(np.mean(Angle_MAE))
    Mag_MAE = np.mean(abs(y_test[:,118:236] - pred[:,118:236]), axis=0)
    print(np.mean(Mag_MAE))

    
    Mag_MAE_all_buses = Mag_MAE_non_PMU
    print(Mag_MAE_all_buses)
    
    Angle_MAE_all_buses = Angle_MAE_non_PMU
    print(Angle_MAE_all_buses)
    
    
    for i in range(pmu_loc_np.shape[0]): # Loop to combine PMU and non_PMU observed variables
        curr_PMU_bus = pmu_loc_np[i]
        curr_PMU_val = Mag_MAE_PMU[i]
        Mag_MAE_all_buses = np.insert(Mag_MAE_all_buses,curr_PMU_bus-1, curr_PMU_val)
        curr_PMU_val = Angle_MAE_PMU[i]
        Angle_MAE_all_buses = np.insert(Angle_MAE_all_buses,curr_PMU_bus-1, curr_PMU_val)
        
        
    
       
    barlist1 = plt.bar(Entire_buses[:,0], Mag_MAE_all_buses, color ='blue',width = 0.8)
    #barlist1[np.asarray(pmu_loc1).astype(np.int64)].set_color('r')
    for i in range(len(pmu_loc1)):
        barlist1[pmu_loc1[i]].set_color('r')
    plt.xlabel('Bus number')
    plt.ylabel('Voltage Magnitude Error (MAE)')
    plt.savefig('Vmag_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    plt.show()
    
    
    barlist1 = plt.bar(Entire_buses[:,0], Angle_MAE_all_buses, color ='blue',width = 0.8)
    #barlist1[np.asarray(pmu_loc1).astype(np.int64)].set_color('r')
    for i in range(len(pmu_loc1)):
        barlist1[pmu_loc1[i]].set_color('r')
    plt.xlabel('Bus number')
    plt.ylabel('Voltage Angle Error (MAE)')
    plt.savefig('Vang_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    plt.show()
    return(Mag_MAE, Angle_MAE)
    
