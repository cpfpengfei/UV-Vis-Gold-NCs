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

N_CALLS = int(sys.argv[1])
print(f"Running hyperparameters optimization for {N_CALLS} iterations!")

# features, labels = get_features_labels()
X_train, Y_train, X_test, Y_test = get_train_test(None, None, random_state = 42, recover = True)
print("Shapes:")
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

def cnn_model(n_input, n_classes, filter1, filter2, filter3, learning_rate):
    model = Sequential()
    model.add(K.layers.Conv1D(filters = filter1, kernel_size = 8, strides=8, padding='same', input_shape=(n_input, 1), activation='relu'))
    model.add(K.layers.Conv1D(filters = filter2, kernel_size = 5, strides=5, padding='same', activation='relu'))
    model.add(K.layers.Conv1D(filters = filter3, kernel_size = 3, strides=3, padding='same', activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(n_classes, activation='softmax'))
    optimizer = K.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mean_absolute_error',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model 

# Set up dimensions --> Edited to integers 
dim_filter1 = Integer(low = 16, high = 200, name='filter1', dtype = int)
dim_filter2 = Integer(low = 16, high = 200, name='filter2', dtype = int)
dim_filter3 = Integer(low = 16, high = 200, name='filter3', dtype = int)
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
dim_batch_size = Integer(low = 10, high = 64, name='batch_size', dtype = int) # NOTE: Must be int so it works for KerasRegressor copy 

dimensions = [dim_filter1, dim_filter2, dim_filter3, dim_learning_rate, dim_batch_size]

print("Number of dimensions:", len(dimensions))

default_parameters = [64, 64, 64, 1e-4, 16]

@use_named_args(dimensions = dimensions)
def fitness(filter1, filter2, filter3, learning_rate, batch_size):
    n_input = 751 # Total number of wavelengths in UV-VIS pattern
    n_classes = 51 # Number of gold nanocluster classes
    EPOCHS = 300

    print("")
    print("Current hyperparams:")
    print(f"filter1: {filter1}")
    print(f"filter2: {filter2}")
    print(f"filter3: {filter3}")
    print(f"learning_rate: {learning_rate}")
    print(f"batch_size: {batch_size}")

    # wrap model with KerasRegressor in order to include epochs and batch size
    model = KerasRegressor(build_fn = lambda: cnn_model(n_input, n_classes, filter1, filter2, filter3, learning_rate),
                          epochs = EPOCHS, 
                           batch_size = batch_size, 
                           verbose = False)
    
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
                        n_calls = N_CALLS, # 50 runs of hopt
                        x0 = default_parameters,
                        verbose = True)

min(gp_result.func_vals)
gp_result.x

titles = ["filter1", "filter2", "filter3", "learning_rate", "batch_size"]
arr = np.asarray(gp_result.x_iters)
results_dict = {}
for i, k in enumerate(titles):
    results_dict[k] = arr[:, i]
params_df = pd.DataFrame.from_dict(results_dict)
params_df['MAE'] = gp_result.func_vals
params_df.to_csv("CNN_forward_hopt_results_210324.csv")
params_df.nsmallest(20, 'MAE')
