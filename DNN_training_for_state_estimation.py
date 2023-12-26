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


def DNN_training_base_topology(X_train, X_val, y_train, y_val, x, y):

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
    model.compile(loss='mean_squared_error', metrics = ['mae'], optimizer='adam')
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.2,patience=10, min_lr=0.0001)
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True)
    
    # history = model.fit(x_train,y_train,verbose=1,epochs=20,validation_split=0.2,callbacks=[ES])
    start = time.time()
    model.fit(X_train,y_train,verbose=1,epochs=4000,validation_data = (X_val,y_val),batch_size=256,callbacks=[checkpoint,reduce_lr])
    end = time.time()
    elapsed_time = end-start
    elapsed_time
    model.save_weights("C:\Downloads\Base_TF_weights_11.h5")
    model.save("C:\Downloads\Saved_models\Base_topology_DNN_TF_11.h5")
    return(model)