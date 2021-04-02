import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

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

features, labels, multiply_by, added = get_features_labels()
print(f"final multiply_by: {multiply_by} -- added: {added}")
reverse_dic = {"multiply_by": [multiply_by], "added": [added]}
pd.DataFrame.from_dict(reverse_dic).to_csv("data/Reverse_Standardized.csv")

# get fixed train tests (forward) and convert to reverse fixed train tests 
X_train_F, Y_train_F, X_test_F, Y_test_F = get_train_test(features, labels, random_state = 42, recover = True)
X_train, Y_train, X_test, Y_test = convert_train_test_reverse(X_train_F, Y_train_F, X_test_F, Y_test_F, multiply_by, added)
print("X Y Shapes:")
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

def cnn_model(n_input, n_classes, filter1, filter2, filter3, learning_rate):
    model = Sequential()
    model.add(K.layers.Conv1D(filters = filter1, kernel_size = 8, strides=8, padding='same', input_shape=(n_input, 1), activation='relu'))
    model.add(K.layers.Conv1D(filters = filter2, kernel_size = 5, strides=5, padding='same', activation='relu'))
    model.add(K.layers.Conv1D(filters = filter3, kernel_size = 3, strides=3, padding='same', activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(n_classes, activation='sigmoid'))
    optimizer = K.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mean_absolute_error',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model 

# Set up dimensions 
dim_filter1 = Integer(low = 16, high = 200, name='filter1', dtype = int)
dim_filter2 = Integer(low = 16, high = 200, name='filter2', dtype = int)
dim_filter3 = Integer(low = 16, high = 200, name='filter3', dtype = int)
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')

dimensions = [dim_filter1,
             dim_filter2,
              dim_filter3,
             dim_learning_rate]

print("Number of dimensions:", len(dimensions))

default_parameters = [64, 64, 64, 1e-4]

@use_named_args(dimensions = dimensions)
def fitness(filter1, filter2, filter3, learning_rate):
    n_input = 51 # Number of gold nanocluster classes 
    n_classes = 751 # Total wavelengths in UV-VIS pattern
    EPOCHS = 300
    BATCH_SIZE = 10
    
    # wrap model with KerasRegressor in order to include epochs and batch size
    model = KerasRegressor(build_fn = lambda: cnn_model(n_input, n_classes, filter1, filter2, filter3, learning_rate),
                          epochs = EPOCHS, 
                           batch_size = BATCH_SIZE, 
                           verbose = False)
    
    print("")
    print("Current hyperparams:")
    print(f"filter1: {filter1}")
    print(f"filter2: {filter2}")
    print(f"filter3: {filter3}")
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

titles = ["filter1", "filter2", "filter3", "learning_rate"]
arr = np.asarray(gp_result.x_iters)
results_dict = {}
for i, k in enumerate(titles):
    results_dict[k] = arr[:, i]
params_df = pd.DataFrame.from_dict(results_dict)
params_df['MAE'] = gp_result.func_vals
params_df.to_csv("Reverse_hopt_results_210205.csv")
params_df.nsmallest(20, 'MAE')
