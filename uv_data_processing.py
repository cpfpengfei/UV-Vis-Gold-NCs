import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
# preprocessing features labels 
def get_features_labels():
    # load training data 
    full_df = pd.read_csv('data/abundance_new.csv')

    # modify the intensity according to the Au number real_intensity = intensity/Au_num
    label_df = full_df.iloc[:, 6:]
    row, col = label_df.shape
    modified_label_df = pd.DataFrame(np.zeros((row, col)))

    Au_i = 0
    for i in range(0, modified_label_df.shape[1]):
        for Au_i in range(modified_label_df.shape[0]):
            modified_label_df.iloc[Au_i, i] = label_df.iloc[Au_i, i]
    # Convert label values into sum = 1
    for i in range(modified_label_df.shape[1]):
        sum_i = modified_label_df.iloc[:, i].sum()
        for j in range(modified_label_df.shape[0]):
            modified_label_df.iloc[j, i] = modified_label_df.iloc[j, i]/sum_i
    label_processed_df = modified_label_df

    # Preprocess feature data: UV-Vis spectrum
    spectrum_df = pd.read_csv("data/uv_new.csv")

    # only take spectrum from 350 nm to 1100 nm
    x_feature = spectrum_df.iloc[151:, 1:254]
    x_feature = x_feature.astype(float)

    # finalise features and labels for forward 
    features = x_feature.values.transpose()
    y_label = label_processed_df.iloc[:,:]
    labels = y_label.values.transpose()

    # just to get the scaler multiplier and added 
    all_uv_values = x_feature.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_all_uv_values = scaler.fit_transform(all_uv_values)
    multiply_by = scaler.scale_[0]
    added = scaler.min_[0]
    print(f"Default multiply_by: {multiply_by}, added: {added}")
    return features, labels, multiply_by, added

def get_train_test(features, labels, random_state = 42, recover = True):
    if not recover:
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        X_train.shape, X_test.shape, len(X_train), len(X_test)
        print("Saving fixed train test...")
        with open("data/X_train.npy", "wb") as f:
            np.save(f, X_train)
        with open("data/Y_train.npy", "wb") as f:
            np.save(f, Y_train)
        with open("data/X_test.npy", "wb") as f:
            np.save(f, X_test)
        with open("data/Y_test.npy", "wb") as f:
            np.save(f, Y_test)
    else: 
        print("Reloading fixed train test...")
        with open("data/X_train.npy", "rb") as f:
            X_train = np.load(f)
        with open("data/Y_train.npy", "rb") as f:
            Y_train = np.load(f)
        with open("data/X_test.npy", "rb") as f:
            X_test = np.load(f)
        with open("data/Y_test.npy", "rb") as f:
            Y_test = np.load(f)
    return X_train, Y_train, X_test, Y_test

# converts fixed forward train tests to reverse train tests 
def convert_train_test_reverse(X_train, Y_train, X_test, Y_test, multiply_by, added):
    Y_train_R = X_train * multiply_by + added
    Y_test_R = X_test * multiply_by + added
    Y_train_R = Y_train_R.reshape(Y_train_R.shape[0], Y_train_R.shape[1])
    Y_test_R = Y_test_R.reshape(Y_test_R.shape[0], Y_test_R.shape[1])
    X_train_R = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)
    X_test_R = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], 1)
    return X_train_R, Y_train_R, X_test_R, Y_test_R

# reverts from scale 0 - 1 back to UV Vis raw data for comparisons and plots
def revert_Y_reverse(Y, multiply_by, added):
    Y_ = Y - added
    Y_actual = Y_ / multiply_by
    return Y_actual
