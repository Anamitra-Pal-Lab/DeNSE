import numpy as np
import matplotlib.pyplot as plt

def Estimation_Error_MAE_viz(X_test, y_test, pred, pmu_loc1):
    
    
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
    
    #%%
    
    # Calculating the MAE of PMU pbserved and unobserved buses separately for magnitude and phase angle variables
    Mag_MAE_PMU = np.mean(abs(y_test[:,pmu_loc_np-1] - pred[:,pmu_loc_np-1]), axis=0)
    Mag_MAE_non_PMU = np.mean(abs(y_test[:,PMU_unobserved_buses-1] - pred[:,PMU_unobserved_buses-1]), axis=0)
    Angle_MAE_PMU = np.mean(abs(y_test[:,pmu_loc_np+118-1] - pred[:,pmu_loc_np+118-1]), axis=0)
    Angle_MAE_non_PMU = np.mean(abs(y_test[:,PMU_unobserved_buses+118-1] - pred[:,PMU_unobserved_buses+118-1]), axis=0)
    
    
    
    #%%
    Mag_MAE_all_buses = Mag_MAE_non_PMU
    Angle_MAE_all_buses = Angle_MAE_non_PMU
    
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
    
    #%%