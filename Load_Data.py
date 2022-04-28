import pandas as pd

def Load_Training_Data(filepath):
    # Reading the current and voltage data
    # Reading the voltages and current measurements to pandas daraframes
    # If - from bus current
    # It - to bus current
    # V - voltage
    # mag - magnitude
    # ang - phase angle  

    df_If_mag = pd.read_csv(filepath+('If_mag_train_T1.csv'), header = None)
    df_If_ang = pd.read_csv(filepath+('If_ang_train_T1.csv'), header = None)
    df_It_mag = pd.read_csv(filepath+('It_mag_train_T1.csv'), header = None)
    df_It_ang = pd.read_csv(filepath+('It_ang_train_T1.csv'), header = None)
    
    df_VDATA_mag = pd.read_csv(filepath+('VDATA_mag_train_T1.csv'), header = None)
    df_VDATA_ang = pd.read_csv(filepath+('VDATA_ang_train_T1.csv'), header = None)
    
    df_From_To_buses_118 = pd.read_csv(filepath+'From_To_buses_118.csv', header = None)
    
    print('Read the train csv files')

    return(df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang, df_From_To_buses_118)



def Load_Testing_Data(filepath):
    # Reading the current and voltage data
    # Reading the voltages and current measurements to pandas daraframes
    # If - from bus current
    # It - to bus current
    # V - voltage
    # mag - magnitude
    # ang - phase angle  

    df_If_mag = pd.read_csv(filepath+('If_mag_test_T1.csv'), header = None)
    df_If_ang = pd.read_csv(filepath+('If_ang_test_T1.csv'), header = None)
    df_It_mag = pd.read_csv(filepath+('It_mag_test_T1.csv'), header = None)
    df_It_ang = pd.read_csv(filepath+('It_ang_test_T1.csv'), header = None)
    
    df_VDATA_mag = pd.read_csv(filepath+('VDATA_mag_test_T1.csv'), header = None)
    df_VDATA_ang = pd.read_csv(filepath+('VDATA_ang_test_T1.csv'), header = None)
   
    print('Read the test csv files')

    return(df_VDATA_mag, df_VDATA_ang, df_If_mag, df_If_ang, df_It_mag, df_It_ang)