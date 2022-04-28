import numpy as np
import pandas as pd

def PMU_current_flow_identifier(pmu_loc1, df_From_To_buses_118):

    From_To_buses_118_np = df_From_To_buses_118.to_numpy()
    From_To_buses_118_np = df_From_To_buses_118.to_numpy() # From and two bus list of branches from Matpower - (Branches x 2)

    # Initial 13 PMU locations in IEEE 118 bus test system
    
    PMU_current_from_bus_branch_num = []
    PMU_current_to_bus_branch_num = []
    
    
    for pmu_location in range(len(pmu_loc1)):
        bus_chosen = pmu_loc1[pmu_location]
        for qf in range(From_To_buses_118_np[:,0].shape[0]):
            if(From_To_buses_118_np[qf,0]==bus_chosen):
              PMU_current_from_bus_branch_num.append(qf)
              #print(str(qf), str(bus_chosen))
            if(From_To_buses_118_np[qf,1]==bus_chosen):
              PMU_current_to_bus_branch_num.append(qf)
              #print(str(From_To_buses_118_np[qf,0]), str(bus_chosen))
    
    
    PMU_current_from_bus_branch_num = np.asarray(PMU_current_from_bus_branch_num)
    From_branches_req = np.unique(PMU_current_from_bus_branch_num)
    From_branches_req = np.sort(From_branches_req)
    
    PMU_current_to_bus_branch_num = np.asarray(PMU_current_to_bus_branch_num)
    To_branches_req = np.unique(PMU_current_to_bus_branch_num)
    To_branches_req = np.sort(To_branches_req)
    
    return(From_branches_req, To_branches_req)


def Generate_noisy_measurements_Gaussian_noise(sigma_mag, sigma_ang, df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118, pmu_loc1, From_branches_req, To_branches_req):
    
    df_VDATA_ang_renamed_whole = df_VDATA_ang.rename(columns={x:y for x,y in zip(df_VDATA_ang.columns,range(0+118,len(df_VDATA_ang.columns)+118))}) 


    df_V_mag_PMU = df_VDATA_mag[df_VDATA_mag.columns[pmu_loc1]]
    df_V_ang_PMU = df_VDATA_ang[df_VDATA_ang.columns[pmu_loc1]]
    
    df_If_mag_PMU = df_If_mag[df_If_mag.columns[From_branches_req]]
    df_If_ang_PMU = df_If_ang[df_If_ang.columns[From_branches_req]]
    df_It_mag_PMU = df_It_mag[df_It_mag.columns[To_branches_req]]
    df_It_ang_PMU = df_It_ang[df_It_ang.columns[To_branches_req]]
    
    offset = 0
    df_V_mag_PMU = df_V_mag_PMU.rename(columns={x:y for x,y in zip(df_V_mag_PMU.columns,range(0+offset,len(df_V_mag_PMU.columns)+offset))}) 
    offset = df_V_mag_PMU.shape[1]
    df_V_ang_renamed = df_V_ang_PMU.rename(columns={x:y for x,y in zip(df_V_ang_PMU.columns,range(0+offset,len(df_V_ang_PMU.columns)+offset))})  
    offset = df_V_mag_PMU.shape[1] + df_V_ang_PMU.shape[1] 
    df_If_mag_renamed = df_If_mag_PMU.rename(columns={x:y for x,y in zip(df_If_mag_PMU.columns,range(0+offset,len(df_If_mag_PMU.columns)+offset))})    
    offset = df_V_mag_PMU.shape[1] + df_V_ang_PMU.shape[1] + df_If_mag_PMU.shape[1]     
    df_If_ang_renamed = df_If_ang_PMU.rename(columns={x:y for x,y in zip(df_If_ang_PMU.columns,range(0+offset,len(df_If_ang_PMU.columns)+offset))})                                                                                         
    offset = df_V_mag_PMU.shape[1] + df_V_ang_PMU.shape[1] + df_If_mag_PMU.shape[1]*2
    df_It_mag_renamed = df_It_mag_PMU.rename(columns={x:y for x,y in zip(df_It_mag_PMU.columns,range(0+offset,len(df_It_mag_PMU.columns)+offset))})    
    offset = df_V_mag_PMU.shape[1] + df_V_ang_PMU.shape[1] + df_If_mag_PMU.shape[1]*2 + df_It_mag_PMU.shape[1]   
    df_It_ang_renamed = df_It_ang_PMU.rename(columns={x:y for x,y in zip(df_It_ang_PMU.columns,range(0+offset,len(df_It_ang_PMU.columns)+offset))})  
    offset = df_V_mag_PMU.shape[1] + df_V_ang_PMU.shape[1] + df_If_mag_PMU.shape[1]*2 + df_It_mag_PMU.shape[1]*2  
    
    
    
    n_Sample_ip = df_V_mag_PMU.shape[0]
    n_feature_ip = df_V_mag_PMU.shape[1]
    df_V_mag_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_mag, columns = df_V_mag_PMU.columns)
    df_V_mag_PMU_noisy = df_V_mag_PMU.add(df_V_mag_PMU_noise, fill_value=0)
    
    
    n_Sample_ip = df_V_ang_PMU.shape[0]
    n_feature_ip = df_V_ang_PMU.shape[1]
    df_V_ang_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_ang, columns = df_V_ang_renamed.columns)
    df_V_ang_PMU_noisy = df_V_ang_renamed.add(df_V_ang_PMU_noise, fill_value=0)
    
    
    
    n_Sample_ip = df_If_mag_PMU.shape[0]
    n_feature_ip = df_If_mag_PMU.shape[1]
    df_If_mag_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_mag, columns = df_If_mag_renamed.columns)
    df_If_mag_PMU_noisy = df_If_mag_renamed.add(df_If_mag_PMU_noise, fill_value=0)
    
    
    n_Sample_ip = df_If_ang_PMU.shape[0]
    n_feature_ip = df_If_ang_PMU.shape[1]
    df_If_ang_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_ang, columns = df_If_ang_renamed.columns)
    df_If_ang_PMU_noisy = df_If_ang_renamed.add(df_If_ang_PMU_noise, fill_value=0)
    
    
    n_Sample_ip = df_It_mag_PMU.shape[0]
    n_feature_ip = df_It_mag_PMU.shape[1]
    df_It_mag_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_mag, columns = df_It_mag_renamed.columns)
    df_It_mag_PMU_noisy = df_It_mag_renamed.add(df_It_mag_PMU_noise, fill_value=0)
    
    
    n_Sample_ip = df_It_ang_PMU.shape[0]
    n_feature_ip = df_It_ang_PMU.shape[1]
    df_It_ang_PMU_noise = pd.DataFrame(np.random.randn(n_Sample_ip, n_feature_ip)*sigma_ang, columns = df_It_ang_renamed.columns)
    df_It_ang_PMU_noisy = df_It_ang_renamed.add(df_It_ang_PMU_noise, fill_value=0)
    
    
    
    # Creating input matrix by concatenating
    df_Input_NN = pd.concat([df_V_mag_PMU_noisy, df_V_ang_PMU_noisy, df_If_mag_PMU_noisy, df_If_ang_PMU_noisy, df_It_mag_PMU_noisy, df_It_ang_PMU_noisy ], axis=1) # Concatenating voltage magnitude and angles to get input
    
    # Creating input matrix by concatenating voltage magnitudes and angles
    df_Output_NN = pd.concat([df_VDATA_mag, df_VDATA_ang_renamed_whole], axis=1) # Concatenating voltage magnitude and angles to get output


    return(df_Input_NN, df_Output_NN)


def Input_Data_Normalization1(df_Input_NN):
    # # Normalizing the input
    cols = list(df_Input_NN.columns)
    df_Input_NN_normalized = pd.DataFrame()
    for col in cols:
        col_zscore = str(col) + '_zscore'
        df_Input_NN_normalized[col_zscore] = (df_Input_NN[col] - df_Input_NN[col].mean())/df_Input_NN[col].std(ddof=0) 
    return(df_Input_NN_normalized)


def Data_Removing_NaNs(df_Input_NN_normalized, df_Output_NN):
    # Removing possible nans from normalized input
    df_Input_NN_normalized = df_Input_NN_normalized.fillna(0)
    df_Output_NN = df_Output_NN.fillna(0)
    return(df_Input_NN_normalized, df_Output_NN)



