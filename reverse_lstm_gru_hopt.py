import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import sys

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import tensorflow.keras as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from tensorflow.python.keras import backend as K2
from skopt.utils import use_named_args

np.random.seed(42)
tf.random.set_seed(42)

from uv_data_processing import *
from datetime import date

CURRENT_DATE = date.today().strftime('%y%m%d')

###########################################

SELECTED_MODEL = sys.argv[1]

features, labels, multiply_by, added = get_features_labels()
print(f"final multiply_by: {multiply_by} -- added: {added}")
reverse_dic = {"multiply_by": [multiply_by], "added": [added]}
pd.DataFrame.from_dict(reverse_dic).to_csv(f"data/Reverse_Standardized_{CURRENT_DATE}.csv")

# get fixed train tests (forward) and convert to reverse fixed train tests 
X_train_F, Y_train_F, X_test_F, Y_test_F = get_train_test(features, labels, random_state = 42, recover = True)
X_train, Y_train, X_test, Y_test = convert_train_test_reverse(X_train_F, Y_train_F, X_test_F, Y_test_F, multiply_by, added)
print("X Y Shapes:")
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

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

# Set up dimensions 

def create_dimensions(MODEL = "LSTM"):
    if SELECTED_MODEL == "LSTM":
        dim_layer1 = Integer(low = 1, high = 20, name='layer1')
        dim_layer2 = Integer(low = 1, high = 20, name='layer2')
        dim_layer3 = Integer(low = 1, high = 20, name='layer3')
        dim_layer4 = Integer(low = 1, high = 20,  name='layer4')
        dim_dropout = Real(low=0.0, high=0.5, prior='uniform', name='dropout')
        dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dimensions = [dim_layer1, dim_layer2, dim_layer3, dim_layer4, dim_dropout, dim_learning_rate]
        default_parameters = [1, 1, 1, 1, 0.0, 1e-2]
    else:
        dim_layer1 = Integer(low = 1, high = 20, name='layer1')
        dim_layer2 = Integer(low = 1, high = 20, name='layer2')
        dim_layer3 = Integer(low = 1, high = 20, name='layer3')
        dim_layer4 = Integer(low = 1, high = 20,  name='layer4')
        dim_dropout = Real(low=0.0, high=0.5, prior='uniform', name='dropout')
        dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dimensions = [dim_layer1, dim_layer2, dim_layer3, dim_layer4, dim_dropout, dim_learning_rate]
        default_parameters = [1, 1, 1, 1, 0.0, 1e-2]
    print("Number of dimensions:", len(dimensions))   
    return dimensions, default_parameters

dimensions, default_parameters = create_dimensions(MODEL = SELECTED_MODEL)
@use_named_args(dimensions = dimensions)
def fitness(layer1, layer2, layer3, layer4, dropout, learning_rate):
    n_input = 51 # Number of gold nanocluster classes 
    n_classes = 751 # Total wavelengths in UV-VIS pattern
    EPOCHS = 300
    BATCH_SIZE = 10
    
    if SELECTED_MODEL == "LSTM":
        # wrap model with KerasRegressor in order to include epochs and batch size
        model = KerasRegressor(build_fn = lambda: lstm_model(n_input, n_classes, layer1, layer2, layer3, layer4, dropout, learning_rate),
                                epochs = EPOCHS, 
                                batch_size = BATCH_SIZE, 
                                verbose = False)
    else:
        model = KerasRegressor(build_fn = lambda: gru_model(n_input, n_classes, layer1, layer2, layer3, layer4, dropout, learning_rate),
                                epochs = EPOCHS, 
                                batch_size = BATCH_SIZE, 
                                verbose = False)
    
    print("")
    print("Current hyperparams:")
    print(f"layer1: {layer1}")
    print(f"layer2: {layer2}")
    print(f"layer3: {layer3}")
    print(f"layer4: {layer4}")
    print(f"dropout: {dropout}")
    print(f"learning_rate: {learning_rate}")
    
    # 5 sets of train validation splits 
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    results = cross_val_score(model, X_train, Y_train, 
                             cv = kfold, 
                             scoring = 'neg_mean_absolute_error',
                             verbose = 1)
    print(results)
    mean_neg_MAE = results.mean()

    K2.clear_session()
    return -mean_neg_MAE

gp_result = gp_minimize(func = fitness,
                        dimensions = dimensions,
                        acq_func = "EI",
                        xi = 0.01,
                        n_calls = 50, # 50 runs of hopt
                        x0 = default_parameters,
                        verbose = True)

min(gp_result.func_vals)
gp_result.x

if SELECTED_MODEL == "LSTM":
    titles = ["layer1", "layer2", "layer3", "layer4", "dropout", "learning_rate"]
    file_prefix = "LSTM"
else:
    titles = ["layer1", "layer2", "layer3", "layer4", "dropout", "learning_rate"]
    file_prefix = "GRU"
arr = np.asarray(gp_result.x_iters)
results_dict = {}
for i, k in enumerate(titles):
    results_dict[k] = arr[:, i]
params_df = pd.DataFrame.from_dict(results_dict)
params_df['MAE'] = gp_result.func_vals
params_df.to_csv(f"{file_prefix}_reverse_hopt_results_{CURRENT_DATE}.csv")
params_df.nsmallest(10, 'MAE')