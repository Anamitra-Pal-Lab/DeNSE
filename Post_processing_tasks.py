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


def Post_processing(y_test, pred):
    Mag_MAE =np.mean(abs(y_test[:,0:118] - pred[:,0:118]), axis=0)*100
    print(np.mean(Mag_MAE))
    Angle_MAE = np.mean(abs(y_test[:,118:236] - pred[:,118:236]), axis=0)
    print(np.mean(Angle_MAE))
    pd.DataFrame(Mag_MAE).to_csv('Mag_MAPE_118.csv')
    pd.DataFrame(Angle_MAE).to_csv('Angle_MAE_118.csv')
              
    barlist1 = plt.bar(Entire_buses[:,0], Mag_MAE , color ='blue',width = 0.8)
    plt.xlabel('Bus number')
    plt.ylabel('Voltage Magnitude Error (MAPE)')
    plt.savefig('Vmag_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    plt.show()
        
        
    barlist1 = plt.bar(Entire_buses[:,0], Angle_MAE, color ='blue',width = 0.8)
    plt.xlabel('Bus number')
    plt.ylabel('Voltage Angle Error (MAE)')
    plt.savefig('Vang_MAE_SE_T1.png',bbox_inches='tight', dpi=300)
    plt.show()
    
    return(Mag_MAE, Angle_MAE)