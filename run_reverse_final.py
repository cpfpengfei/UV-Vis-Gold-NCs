import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import tensorflow.keras as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

np.random.seed(42)
tf.random.set_seed(42)

from uv_data_processing import *

CURRENT_DATE = date.today().strftime('%y%m%d')

##############################################

def get_pearson_R(Y_test, Y_pred):
    list_of_Rs = []
    for i in range(len(Y_test)):
        R = np.corrcoef(Y_test[i], Y_pred[i])[0, 1]
        list_of_Rs.append(R)
    mean_R = np.mean(list_of_Rs)
    min_max_R = (np.min(list_of_Rs), np.max(list_of_Rs))
    min_max_R_i = (list_of_Rs.index(np.min(list_of_Rs)), list_of_Rs.index(np.max(list_of_Rs)))
    return list_of_Rs, mean_R, min_max_R, min_max_R_i

def cnn_model(n_input, n_classes, filter1, filter2, filter3, learning_rate):
    model = Sequential()
    model.add(K.layers.Conv1D(filters = filter1, kernel_size = 8, strides=8, padding='same', input_shape=(n_input, 1), activation='relu'))
    model.add(K.layers.Conv1D(filters = filter2, kernel_size = 5, strides=5, padding='same', activation='relu'))
    model.add(K.layers.Conv1D(filters = filter3, kernel_size = 3, strides=3, padding='same', activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(n_classes, activation='sigmoid')) # sigmoid for non sum 1
    optimizer = K.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mean_absolute_error',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model 

def lstm_model(n_input, n_classes, layer1, layer2, layer3, layer4, dropout, learning_rate):
    model = Sequential()
    model.add(LSTM(layer1, return_sequences=True, input_shape=(n_input,1)))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(LSTM(layer2, return_sequences=True))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(LSTM(layer3, return_sequences=True))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(LSTM(layer4, return_sequences=False))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation = "sigmoid")) # sigmoid for non sum 1
    optimizer = K.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mean_absolute_error',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model 

def gru_model(n_input, n_classes, layer1, layer2, layer3, layer4, dropout, learning_rate):
    model = Sequential()
    model.add(GRU(layer1, return_sequences=True, input_shape=(n_input,1)))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(GRU(layer2, return_sequences=True))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(GRU(layer3, return_sequences=True))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(GRU(layer4, return_sequences=False))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation = "sigmoid")) # sigmoid for non sum 1
    optimizer = K.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mean_absolute_error',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model 

##############################################
print("PREPROCESSING FIXED TRAIN TEST DATA...")
print("")
features, labels, multiply_by, added = get_features_labels()
print(f"final multiply_by: {multiply_by} -- added: {added}")

# get fixed train tests (forward) and convert to reverse fixed train tests 
X_train_F, Y_train_F, X_test_F, Y_test_F = get_train_test(features, labels, random_state = 42, recover = True)
X_train, Y_train, X_test, Y_test = convert_train_test_reverse(X_train_F, Y_train_F, X_test_F, Y_test_F, multiply_by, added)
print("X Y Shapes:")
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

Y_test_actual = revert_Y_reverse(Y_test, multiply_by, added) # revert back UV Vis
actual_UVvis = {i : Y_test_actual[i] for i in range(len(X_test))}
pd.DataFrame.from_dict(actual_UVvis).to_csv(f"Reverse Actual UV-Vis Test Sample_{CURRENT_DATE}.csv")

MODEL_PERFORMANCES = {
    "1DCNN" : [],
    "LSTM" : [],
    "GRU" : []
}

n_input = 51 # Number of gold nanocluster classes 
n_classes = 751 # Total wavelengths in UV-VIS pattern
EPOCHS = 300
BATCH_SIZE = 10

cnn_hopt_results = pd.read_csv("FINAL_MODELS_STORE/CNN_reverse_hopt_results_210205.csv")
cnn_hyperparams = list(cnn_hopt_results.nsmallest(1, "MAE").values[0])[1:]
lstm_hopt_results = pd.read_csv("FINAL_MODELS_STORE/LSTM_reverse_hopt_results_210401.csv")
lstm_hyperparams = list(lstm_hopt_results.nsmallest(1, "MAE").values[0])[1:]
gru_hopt_results = pd.read_csv("FINAL_MODELS_STORE/GRU_reverse_hopt_results_210401.csv")
gru_hyperparams = list(gru_hopt_results.nsmallest(1, "MAE").values[0])[1:]

print("1D CNN HYPERPARAMS: ", hyperparams)
print("")

print("RUNNING 1D CNN MODEL...")
print("")
model = cnn_model(n_input, n_classes, 
                int(hyperparams[0]), 
                int(hyperparams[1]), 
                int(hyperparams[2]), 
                hyperparams[3])
model.fit(X_train, Y_train, 
            batch_size = BATCH_SIZE, 
            epochs = EPOCHS, 
            verbose = 0,
            validation_data=(X_test, Y_test))

Y_pred = model.predict(X_test)
print("Y_pred shape:", Y_pred.shape)
Y_pred_actual = revert_Y_reverse(Y_pred, multiply_by, added) # revert back UV Vis
print("Y_pred_actual shape:", Y_pred_actual.shape)

MAE = mean_absolute_error(Y_test, Y_pred)
_, mean_R, _, _ = get_pearson_R(Y_test_actual, Y_pred_actual)

MODEL_PERFORMANCES["1DCNN"].append(MAE)
MODEL_PERFORMANCES["1DCNN"].append(mean_R)
print("1DCNN Test MAE:", MAE, "| Test R:", mean_R)
model.save(f"Rev_1DCNN_1")

predicted_UVVis = {i : Y_pred_actual[i] for i in range(len(X_test))}
pd.DataFrame.from_dict(predicted_UVVis).to_csv(f"Reverse 1D CNN Predicted UV-Vis Test Sample_{CURRENT_DATE}.csv")

print("SAVED 1D CNN MODEL + PERFORMANCE + TEST PREDICTIONS ...")
print("")
del model 

print("RUNNING LSTM MODEL...")
print("")
model = lstm_model(n_input, n_classes, 
                layer1 = int(lstm_hyperparams[0]), 
                layer2 = int(lstm_hyperparams[1]), 
                layer3 = int(lstm_hyperparams[2]), 
                layer4 = int(lstm_hyperparams[3]), 
                dropout = lstm_hyperparams[4],
                learning_rate = lstm_hyperparams[5])
model.fit(X_train, Y_train,
            batch_size= BATCH_SIZE, 
            epochs= EPOCHS, 
            verbose= 0, 
            validation_data=(X_test, Y_test))

Y_pred = model.predict(X_test)
Y_pred_actual = revert_Y_reverse(Y_pred, multiply_by, added) # revert back UV Vis

MAE = mean_absolute_error(Y_test, Y_pred)
_, mean_R, _, _ = get_pearson_R(Y_test_actual, Y_pred_actual)

MODEL_PERFORMANCES["LSTM"].append(MAE)
MODEL_PERFORMANCES["LSTM"].append(mean_R)
print("LSTM Test MAE:", MAE, "| Test R:", mean_R)
model.save(f"Rev_LSTM_1")

predicted_UVVis = {i : Y_pred_actual[i] for i in range(len(X_test))}
pd.DataFrame.from_dict(predicted_UVVis).to_csv(f"Reverse LSTM Predicted UV-Vis Test Sample_{CURRENT_DATE}.csv")

print("SAVED LSTM MODEL + PERFORMANCE + TEST PREDICTIONS ...")
print("")
del model 


print("RUNNING GRU MODEL...")
print("")
model = gru_model(n_input, n_classes, 
                layer1 = int(gru_hyperparams[0]), 
                layer2 = int(gru_hyperparams[1]), 
                layer3 = int(gru_hyperparams[2]), 
                layer4 = int(gru_hyperparams[3]), 
                dropout = gru_hyperparams[4],
                learning_rate = gru_hyperparams[5])
model.fit(X_train, Y_train,
            batch_size= BATCH_SIZE, 
            epochs= EPOCHS, 
            verbose= 0, 
            validation_data=(X_test, Y_test))

Y_pred = model.predict(X_test)
Y_pred_actual = revert_Y_reverse(Y_pred, multiply_by, added) # revert back UV Vis

MAE = mean_absolute_error(Y_test, Y_pred)
_, mean_R, _, _ = get_pearson_R(Y_test_actual, Y_pred_actual)

MODEL_PERFORMANCES["GRU"].append(MAE)
MODEL_PERFORMANCES["GRU"].append(mean_R)
print("GRU Test MAE:", MAE, "| Test R:", mean_R)
model.save(f"Rev_GRU_1")

predicted_UVVis = {i : Y_pred_actual[i] for i in range(len(X_test))}
pd.DataFrame.from_dict(predicted_UVVis).to_csv(f"Reverse GRU Predicted UV-Vis Test Sample_{CURRENT_DATE}.csv")

print("SAVED GRU MODEL + PERFORMANCE + TEST PREDICTIONS ...")
print("")
del model 

pd.DataFrame.from_dict(MODEL_PERFORMANCES).to_csv(f"Reverse Model Performances (Fixed Test)_{CURRENT_DATE}.csv")
print("SAVED ALL PERFORMANCES TO CSV...")
print("")

#################################
# Do on incremental train set 

partition_sizes = []
for i in range(1, 227//5): # increment of 5 each time 
    partition_sizes.append(i*5)
partition_sizes.append(227)

MAE_list = []
R_list = []

for train_size in partition_sizes:
    current_X_train = X_train[:train_size]
    current_Y_train = Y_train[:train_size]
    
    print("Current train set shape:")
    print(current_X_train.shape, current_Y_train.shape)

    model = cnn_model(n_input, n_classes, 
                    int(hyperparams[0]), 
                    int(hyperparams[1]), 
                    int(hyperparams[2]), 
                    hyperparams[3])
    model.fit(current_X_train, current_Y_train, 
                batch_size = BATCH_SIZE, 
                epochs = EPOCHS, 
                verbose = 0,
                validation_data=(X_test, Y_test))

    Y_pred = model.predict(X_test)
    Y_pred_actual = revert_Y_reverse(Y_pred, multiply_by, added) # revert back UV Vis
    Y_test_actual = revert_Y_reverse(Y_test, multiply_by, added) # revert back UV Vis

    MAE = mean_absolute_error(Y_test, Y_pred)
    _, mean_R, _, _ = get_pearson_R(Y_test_actual, Y_pred_actual)

    MAE_list.append(MAE)
    R_list.append(mean_R)
    del model

train_set_vary_df = {
    "partition_sizes" : partition_sizes,
    "MAE" : MAE_list,
    "R" : R_list,
}
pd.DataFrame.from_dict(train_set_vary_df).to_csv(f"Reverse 1D CNN training set size results_{CURRENT_DATE}.csv")
print("")
print("EVERYTHING DONE!")
print("#################################")

#################################