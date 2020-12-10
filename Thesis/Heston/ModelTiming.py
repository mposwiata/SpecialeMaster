import numpy as np
import tensorflow as tf
import itertools
import joblib
import tensorflow as tf
from keras.models import load_model
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

from Thesis.Heston import AndersenLake as al, HestonModel as hm, DataGeneration as dg, ModelGenerator as mg, ModelTesting as mt
from Thesis.misc import VanillaOptions as vo

def timing(model_string : str) -> float:
    some_model = hm.HestonClass(100, 0.04, 2, 0.04, 0.5, -0.7, 0.05)
    some_model_array = np.array((100, 0.04, 2, 0.04, 0.5, -0.7, 0.05))
    option = mt.option_input()
    some_option_list = np.array([])
    for some_option in option:
        some_option_list = np.append(some_option_list, vo.EUCall(some_option[0], some_option[1]))

    model_string = "standardize_single_5_1000.h5"

    model_string = ' '.join(glob.glob("Models5/*/"+model_string))
    model = load_model(model_string)
    model_folder = model_string[:model_string.rfind("/") + 1]
    if os.path.exists(model_folder+"/norm_feature.pkl"):
        norm_feature = joblib.load(model_folder+"norm_feature.pkl")
        normal_in = True
    else:
        normal_in = False

    if os.path.exists(model_folder+"/norm_labels.pkl"):
        norm_labels = joblib.load(model_folder+"norm_labels.pkl")
        normal_out = True
    else:
        normal_out = False

    if (model_string.find("single") != -1): # single output
        predictions = np.zeros(np.shape(option)[0])
        test_grid = np.zeros((25, 9))
        for i in range(np.shape(option)[0]):
            test_single_input = np.concatenate((some_model_array, option[i]), axis=None)
            test_single_input = np.reshape(test_single_input, (1, -1))
            if normal_in:
                test_single_input = norm_feature.transform(test_single_input)
            test_grid[i] = test_single_input
    elif (model_string.find("mat") != -1): # single output
        predictions = np.zeros(np.shape(option)[0])
        test_grid = np.zeros((5, 8))
        for i in range(5):
            test_mat_input = np.concatenate((some_model_array, option[i*5][0]), axis=None)
            test_mat_input = np.reshape(test_mat_input, (1, -1))
            if normal_in:
                test_mat_input = norm_feature.transform(test_mat_input)
            test_grid[i] = test_mat_input
    else: # we have a grid
        if normal_in:
            some_model_array = norm_feature.transform(some_model_array)
        test_grid = some_model_array

    inp_tensor = tf.convert_to_tensor(test_grid)

    start_time = time.time()
    for i in range(100):
        if normal_out:
            predictions = norm_labels.inverse_transform(model(test_grid))
        else:
            predictions = model(test_grid)
    stop_time = time.time()
    return (stop_time - start_time)/100

if __name__ == "__main__":
    model_string1 = "standardize_single_5_1000.h5"
    time1 = timing(model_string1)
    print(model_string1+" took: ", time1)

    model_string2 = "standardize_mat_5_1000.h5"
    time2 = timing(model_string2)
    print(model_string2+" took: ", time2)

    model_string3 = "standardize_5_1000.h5"
    time3 = timing(model_string2)
    print(model_string3+" took: ", time3)
