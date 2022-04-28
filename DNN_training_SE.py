from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

def DNN_training(X_train, X_val, y_train, y_val, n_epochs, n_neurons_per_layer, n_hidden_layers):
    # Build the neural network
    model = Sequential()
    model.add(Dense(n_neurons_per_layer, input_dim=X_train.shape[1], activation='relu')) # Hidden 1
    for i in range(n_hidden_layers):
        model.add(Dense(n_neurons_per_layer, activation='relu')) # Hidden 2
    model.add(Dense(y_train.shape[1])) # Output 
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=300, verbose=1, mode='auto', restore_best_weights=True)
    model.fit(X_train,y_train,validation_data=(X_val,y_val),callbacks=[monitor],verbose=2,epochs=n_epochs)
    return(model)



