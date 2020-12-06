import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import joblib
import glob
import time
import timeit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras import backend as k
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from multiprocessing import Pool, cpu_count
from scipy.special import ndtr
from scipy.stats import norm
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import MonteCarlo as mc, AndersenLake as al, HestonModel as hm, NNModelGenerator as mg, ModelGenerator
from Thesis.misc import VanillaOptions as vo
from Thesis import NeuralNetworkGenerator as nng
from Thesis.BlackScholes import BlackScholes as bs

def lr_schedule(epoch, rate):
    lower_lr = 1e-4
    upper_lr = lower_lr * 100
    no_epochs = 100
    peak_epoch = 45
    if epoch <= peak_epoch:
        lr = lower_lr + epoch / peak_epoch * (upper_lr - lower_lr)
    elif peak_epoch < epoch < peak_epoch * 2:
        lr = upper_lr - (epoch - peak_epoch) / peak_epoch * (upper_lr - lower_lr)
    else:
        lr = lower_lr * (1 - (epoch - 2 * peak_epoch) / (no_epochs - 2 * peak_epoch)) * (1 - 1 / 10)

    return lr

def plot_func_training(x_axis : np.array, easy_output : list, hard_output : list, title : str):
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    easy_ax = plt.subplot(121)
    hard_ax = plt.subplot(122)
    easy_ax.plot(x_axis, easy_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    easy_ax.plot(x_axis, easy_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    easy_ax.plot(x_axis, easy_output[2], 'y-', alpha=0.5, label="Andersen Lake, multi")
    easy_ax.plot(x_axis, easy_output[3], 'k-', alpha=0.5, label="Monte Carlo, multi")
    hard_ax.plot(x_axis, hard_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    hard_ax.plot(x_axis, hard_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    hard_ax.plot(x_axis, hard_output[2], 'y-', alpha=0.5, label="Andersen Lake, multi")
    hard_ax.plot(x_axis, hard_output[3], 'k-', alpha=0.5, label="Monte Carlo, multi")
    handles, labels = easy_ax.get_legend_handles_labels()
    fig.suptitle(title,fontsize=20)
    easy_ax.set_xlabel("Spot")
    hard_ax.set_xlabel("Spot")
    easy_ax.set_title("Easy case")
    hard_ax.set_title("Hard case")
    fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.15)
    fig.legend(handles, labels, loc="lower center", ncol = 2, fontsize=15)
    plt.savefig("al_mc_"+title.replace(" ", "_").replace(",","")+".png")
    plt.close()

def plot_func(x_axis : np.array, plot_data : dict, title : str):
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    easy_ax = plt.subplot(121)
    hard_ax = plt.subplot(122)
    color=iter(plt.cm.tab10(np.linspace(0,1,len(plot_data))))
    for key in plot_data:
        if key == "Andersen Lake":
            c = 'black'
        else:
            c = next(color)
        easy_ax.plot(x_axis, plot_data[key][0], color = c, label = key)
        hard_ax.plot(x_axis, plot_data[key][1], color = c, label = key)
        
    handles, labels = easy_ax.get_legend_handles_labels()
    fig.suptitle(title,fontsize=20)
    easy_ax.set_xlabel("Spot")
    hard_ax.set_xlabel("Spot")
    easy_ax.set_title("Easy case")
    hard_ax.set_title("Hard case")
    fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.15)
    fig.legend(handles, labels, loc="lower center", ncol = 3, fontsize=15)
    plt.savefig("al_mc_"+title.replace(" ", "_").replace(",","")+".png")
    plt.close()

def generate_network(X, Y):
    # Modelling
    adam = Adam()

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
    ]
    model = nng.NN_generator(4, 50, np.shape(X)[1], np.shape(Y)[1])

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    model.fit(X, Y, epochs=100, batch_size=10, verbose = 2, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)

    return model

def generate_multi_network(X, Y):
    # Modelling
    adam = Adam()

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
    ]
    model = nng.NN_generator(4, 50, np.shape(X)[1], np.shape(Y)[1])

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    model.fit(X, Y, epochs=100, batch_size=128, verbose = 2, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)

    return model

def generate_predictions(test_x, model, norm_feature, norm_labels):
    ### Derivatives
    inp_tensor = tf.convert_to_tensor(test_x)

    ### Andersen Lake model
    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        with tf.GradientTape() as tape2:
            tape2.watch(inp_tensor)
            predict = model(inp_tensor)
        grads = tape2.gradient(predict, inp_tensor)

    grads2 = tape.gradient(grads, inp_tensor) * (norm_labels.data_max_ - norm_labels.data_min_) / (np.sqrt(norm_feature.var_[0]) ** 2)

    grads = grads * (norm_labels.data_max_ - norm_labels.data_min_) / np.sqrt(norm_feature.var_[0])

    predict = norm_labels.inverse_transform(predict)

    return predict, grads, grads2

def calc_prices(spot : float, epsilon : float):
    vol1 = 0.04
    vol2 = 0.01
    kappa1 = 2
    kappa2 = 0.1
    theta1 = 0.04
    theta2 = 0.01
    epsilon1 = 0.5
    epsilon2 = 2
    rho1 = -0.7
    rho2 = 0.8
    rate = 0.05

    tau = 1.005 #set to match option data
    strike = 100

    some_option = vo.EUCall(tau, strike)

    some_model = hm.HestonClass(spot, vol1, kappa1, theta1, epsilon, rho1, rate) # case 1
    some_model2 = hm.HestonClass(spot, vol2, kappa2, theta2, epsilon, rho2, rate)
    al1 = al.Andersen_Lake(some_model, some_option)
    mc1 = mc.Heston_monte_carlo(some_model, some_option, 10000)
    al2 = al.Andersen_Lake(some_model2, some_option)
    mc2 = mc.Heston_monte_carlo(some_model2, some_option, 10000)

    return al1, mc1, al2, mc2

def model_grads(model_string : str, easy_case : np.ndarray, hard_case : np.ndarray, option : vo.VanillaOption) -> dict:
    model = load_model(model_string)
    model_folder = model_string[:model_string.rfind("/") + 1]
    norm_feature = joblib.load(model_folder+"norm_feature.pkl")

    if os.path.exists(model_folder+"/norm_labels.pkl"):
        norm_labels = joblib.load(model_folder+"norm_labels.pkl")
        normal_out = True
    else:
        normal_out = False

    if (model_string.find("mat") != -1):
        option_no = 2
        mat = np.reshape(np.repeat(1.005, np.shape(easy_case)[0]), (-1, 1))
        easy_case = np.concatenate((easy_case, mat), axis = 1)
        hard_case = np.concatenate((hard_case, mat), axis = 1)
    elif (model_string.find("single") != -1):
        option_no = 0
        mat = np.reshape(np.repeat(1.005, np.shape(easy_case)[0]), (-1, 1))
        strike = np.reshape(np.repeat(100, np.shape(easy_case)[0]), (-1, 1))
        easy_case = np.concatenate((easy_case, mat, strike), axis = 1)
        hard_case = np.concatenate((hard_case, mat, strike), axis = 1)
    else:
        option_no = 12

    if isinstance(norm_feature, MinMaxScaler):
        grads_bot = (norm_feature.data_max_[0] - norm_feature.data_min_[0])
        grads2_bot = ((norm_feature.data_max_[0] - norm_feature.data_min_[0]) ** 2)
    elif isinstance(norm_feature, StandardScaler):
        grads_bot = np.sqrt(norm_feature.var_[0])
        grads2_bot = (np.sqrt(norm_feature.var_[0]) ** 2)
    else:
        grads_bot = 1
        grads2_bot = 1
    if normal_out:
        if isinstance(norm_labels, MinMaxScaler):
            grads_top = (norm_labels.data_max_[option_no] - norm_labels.data_min_[option_no])
        elif isinstance(norm_labels, StandardScaler):
            grads_top = np.sqrt(norm_labels.var_[option_no])
    else:
        grads_top = 1

    grads_scale = grads_top / grads_bot
    grads2_scale = grads_top / grads2_bot

    inp_tensor_easy = tf.convert_to_tensor(norm_feature.transform(easy_case))
    inp_tensor_hard = tf.convert_to_tensor(norm_feature.transform(hard_case))

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_easy)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_easy)
            predict_easy = model(inp_tensor_easy)[:,option_no]
        grads_easy = tape2.gradient(predict_easy, inp_tensor_easy)[:,0]
    
    grads2_easy = tape.gradient(grads_easy, inp_tensor_easy).numpy() 
    grads2_easy = grads2_easy[:,0] * grads2_scale
    grads_easy = grads_easy.numpy() * grads_scale
    try:
        predict_easy = norm_labels.inverse_transform(model(inp_tensor_easy))[:,option_no]
    except:
        predict_easy = model(inp_tensor_easy)[:,option_no]

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_hard)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_hard)
            predict_hard = model(inp_tensor_hard)[:,option_no]
        grads_hard = tape2.gradient(predict_hard, inp_tensor_hard)[:,0]
    
    grads2_hard = tape.gradient(grads_hard, inp_tensor_hard).numpy()
    grads2_hard = grads2_hard[:,0] * grads2_scale
    grads_hard = grads_hard.numpy() * grads_scale
    try:
        predict_hard = norm_labels.inverse_transform(model(inp_tensor_hard))[:,option_no]
    except:
        predict_hard = model(inp_tensor_hard)[:,option_no]

    if (model_string.find("price") == -1): # we are modelling implied vol
        spot_plot = easy_case[:,0]
        rate = easy_case[0,6]
        delta_easy = np.zeros(len(spot_plot))
        price_easy = np.zeros(len(spot_plot))
        gamma_easy = np.zeros(len(spot_plot))
        delta_hard = np.zeros(len(spot_plot))
        price_hard = np.zeros(len(spot_plot))
        gamma_hard = np.zeros(len(spot_plot))

        for i in range(len(spot_plot)):
            model_bs_easy = bs.BlackScholesForward(spot_plot[i], predict_easy[i], rate)
            model_bs_hard = bs.BlackScholesForward(spot_plot[i], predict_hard[i], rate)
            price_easy[i] = model_bs_easy.BSFormula(option)
            delta_easy[i] = model_bs_easy.delta_grads(option, grads_easy[i])
            gamma_easy[i] = model_bs_easy.gamma_grads(option, grads_easy[i], grads2_easy[i])
            
            price_hard[i] = model_bs_hard.BSFormula(option)
            delta_hard[i] = model_bs_hard.delta_grads(option, grads_hard[i])
            gamma_hard[i] = model_bs_hard.gamma_grads(option, grads_hard[i], grads2_hard[i])
    else:
        price_easy = predict_easy
        delta_easy = grads_easy
        gamma_easy = grads2_easy
        
        price_hard = predict_hard
        delta_hard = grads_hard
        gamma_hard = grads2_hard

    return_dict = {
        "pred" : [price_easy, price_hard],
        "delta" : [delta_easy, delta_hard],
        "gamma" : [gamma_easy, gamma_hard]
    }

    return return_dict

def mc_price_grads(model : str, easy_case : np.ndarray, hard_case : np.ndarray) -> dict:
    mc_price_model = load_model(model)
    mc_norm_feature = joblib.load("Models4/Heston_input_scale.pkl")
    mc_norm_labels_price = joblib.load(model[:model.rfind("/")+1]+"price_scale.pkl")

    inp_tensor_easy_mc = tf.convert_to_tensor(mc_norm_feature.transform(easy_case))
    inp_tensor_hard_mc = tf.convert_to_tensor(mc_norm_feature.transform(hard_case))

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_easy_mc)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_easy_mc)
            predict_easy_mc = mc_price_model(inp_tensor_easy_mc)[:,12]
        grads_easy_mc = tape2.gradient(predict_easy_mc, inp_tensor_easy_mc)[:,0]
    
    grads2_easy_mc = tape.gradient(grads_easy_mc, inp_tensor_easy_mc).numpy() 
    grads2_easy_mc = grads2_easy_mc[:,0] * np.sqrt(mc_norm_labels_price.var_[12]) / (np.sqrt(mc_norm_feature.var_[0]) ** 2)
    grads_easy_mc = grads_easy_mc.numpy() * np.sqrt(mc_norm_labels_price.var_[12]) / np.sqrt(mc_norm_feature.var_[0])
    price_predictions_easy_mc = mc_norm_labels_price.inverse_transform(mc_price_model(inp_tensor_easy_mc))[:,12]

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_hard_mc)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_hard_mc)
            predict_hard_mc = mc_price_model(inp_tensor_hard_mc)[:,12]
        grads_hard_mc = tape2.gradient(predict_hard_mc, inp_tensor_hard_mc)[:,0]
    
    grads2_hard_mc = tape.gradient(grads_hard_mc, inp_tensor_hard_mc).numpy()
    grads2_hard_mc = grads2_hard_mc[:,0] * np.sqrt(mc_norm_labels_price.var_[12]) / (np.sqrt(mc_norm_feature.var_[0]) ** 2)
    grads_hard_mc = grads_hard_mc.numpy() * np.sqrt(mc_norm_labels_price.var_[12]) / np.sqrt(mc_norm_feature.var_[0])
    price_predictions_hard_mc = mc_norm_labels_price.inverse_transform(mc_price_model(inp_tensor_hard_mc))[:,12]

    return_dict = {
        "pred" : [price_predictions_easy_mc, price_predictions_hard_mc],
        "delta" : [grads_easy_mc, grads_hard_mc],
        "gamma" : [grads2_easy_mc, grads2_hard_mc]
    }

    return return_dict

def timing(model_string : str, easy_case : np.ndarray, hard_case : np.ndarray) -> dict:
    model = load_model(model_string)
    model_folder = model_string[:model_string.rfind("/") + 1]
    norm_feature = joblib.load(model_folder+"norm_feature.pkl")

    if os.path.exists(model_folder+"/norm_labels.pkl"):
        norm_labels = joblib.load(model_folder+"norm_labels.pkl")
        normal_out = True
    else:
        normal_out = False

    if isinstance(norm_feature, MinMaxScaler):
        grads_bot = (norm_feature.data_max_[0] - norm_feature.data_min_[0])
        grads2_bot = ((norm_feature.data_max_[0] - norm_feature.data_min_[0]) ** 2)
    elif isinstance(norm_feature, StandardScaler):
        grads_bot = np.sqrt(norm_feature.var_[0])
        grads2_bot = (np.sqrt(norm_feature.var_[0]) ** 2)
    else:
        grads_bot = 1
        grads2_bot = 1
    if normal_out:
        if isinstance(norm_labels, MinMaxScaler):
            grads_top = (norm_labels.data_max_[12] - norm_labels.data_min_[12])
        elif isinstance(norm_labels, StandardScaler):
            grads_top = np.sqrt(norm_labels.var_[12])
    else:
        grads_top = 1

    grads_scale = grads_top / grads_bot
    grads2_scale = grads_top / grads2_bot

    inp_tensor_easy = tf.convert_to_tensor(norm_feature.transform(easy_case))
    inp_tensor_hard = tf.convert_to_tensor(norm_feature.transform(hard_case))
    if normal_out:
        easy_start = time.time()
        for i in range(100):
            norm_labels.inverse_transform(model(inp_tensor_easy))
        easy_time = time.time() - easy_start

        hard_start = time.time()
        for i in range(100):
            norm_labels.inverse_transform(model(inp_tensor_hard))
        hard_time = time.time() - hard_start
    else:
        easy_start = time.time()
        for i in range(100):
            model(inp_tensor_easy)
        easy_time = time.time() - easy_start

        hard_start = time.time()
        for i in range(100):
            model(inp_tensor_hard)
        hard_time = time.time() - hard_start

    return easy_time / 100, hard_time / 100

if __name__ == "__main__":
    ### Generating input data
    spot = np.linspace(start = 50, stop = 150, num = 1000)
    vol1 = 0.04
    vol2 = 0.01
    kappa1 = 2
    kappa2 = 0.1
    theta1 = 0.04
    theta2 =  0.01
    epsilon1 = 0.5
    epsilon2 = 2
    epsilon = np.linspace(start = 0.5, stop = 2, num = 10)
    rho1 = -0.7
    rho2 = 0.8
    rate = 0.05
    input_array = np.array(list(itertools.product(spot, epsilon)))

    tau = 1.005 #set to match option data
    strike = 100

    some_option = vo.EUCall(tau, strike)

    spot = np.reshape(spot, (-1, 1))

    ### Single data
    al_output1 = np.loadtxt("Data/al_output1.csv", delimiter=",")
    mc_output1 = np.loadtxt("Data/mc_output1.csv", delimiter=",")
    al_output2 = np.loadtxt("Data/al_output2.csv", delimiter=",")
    mc_output2 = np.loadtxt("Data/mc_output2.csv", delimiter=",")

    al_output1 = np.reshape(al_output1, (-1, 1))
    mc_output1 = np.reshape(mc_output1, (-1, 1))
    al_output2 = np.reshape(al_output2, (-1, 1))
    mc_output2 = np.reshape(mc_output2, (-1, 1))

    ### Multi data
    al_output_multiple_1 = np.loadtxt("Data/al_output_multiple_1.csv", delimiter=",")
    mc_output_multiple_1 = np.loadtxt("Data/mc_output_multiple_1.csv", delimiter=",")
    al_output_multiple_2 = np.loadtxt("Data/al_output_multiple_2.csv", delimiter=",")
    mc_output_multiple_2 = np.loadtxt("Data/mc_output_multiple_2.csv", delimiter=",")

    al_output_multiple_1 = np.reshape(al_output_multiple_1, (-1, 1))
    mc_output_multiple_1 = np.reshape(mc_output_multiple_1, (-1, 1))
    al_output_multiple_2 = np.reshape(al_output_multiple_2, (-1, 1))
    mc_output_multiple_2 = np.reshape(mc_output_multiple_2, (-1, 1))

    ### Plotting training data
    easy_index = input_array[:,1] == 0.5
    hard_index = input_array[:,1] == 2

    training_data = {
        "Andersen Lake" : [al_output1, al_output2],
        "Monte Carlo" : [mc_output1, mc_output2],
        "Andersen Lake, multi" : [al_output_multiple_1[easy_index], al_output_multiple_2[hard_index]],
        "Monte Carlo, multi" : [mc_output_multiple_1[easy_index], mc_output_multiple_2[hard_index]]
    }
    
    plot_func(spot, training_data, "Training data")

    ### Ready for NN
    norm_features = StandardScaler()
    norm_features_multiple = StandardScaler()
    X = norm_features.fit_transform(spot)
    X_multi = norm_features_multiple.fit_transform(input_array)

    norm_labels_al1 = MinMaxScaler()
    Y_al1 = norm_labels_al1.fit_transform(al_output1)
    norm_labels_mc1 = MinMaxScaler()
    Y_mc1 = norm_labels_mc1.fit_transform(mc_output1)
    norm_labels_al2 = MinMaxScaler()
    Y_al2 = norm_labels_al2.fit_transform(al_output2)
    norm_labels_mc2 = MinMaxScaler()
    Y_mc1 = norm_labels_mc2.fit_transform(mc_output2)

    al_models1 = load_model("Models5/MC_poc/al_model1.h5")
    #al_model1 = generate_network(X, Y_al1)
    #al_model1.save("Models5/MC_poc/al_model1.h5")
    mc_model1 = load_model("Models5/MC_poc/mc_model1.h5")
    #mc_model1 = generate_network(X, Y_mc1)
    #mc_model1.save("Models5/MC_poc/mc_model1.h5")
    al_model2 = load_model("Models5/MC_poc/al_model2.h5")
    #al_model2 = generate_network(X, Y_al2)
    #al_model2.save("Models5/MC_poc/al_model2.h5")
    mc_model2 = load_model("Models5/MC_poc/mc_model2.h5")
    #mc_model2 = generate_network(X, Y_mc1)
    #mc_model2.save("Models5/MC_poc/mc_model2.h5")

    norm_labels_multi_al1 = MinMaxScaler()
    Y_multi_al1 = norm_labels_multi_al1.fit_transform(al_output_multiple_1)
    norm_labels_multi_mc1 = MinMaxScaler()
    Y_multi_mc1 = norm_labels_multi_mc1.fit_transform(mc_output_multiple_1)
    norm_labels_multi_al2 = MinMaxScaler()
    Y_multi_al2 = norm_labels_multi_al2.fit_transform(al_output_multiple_2)
    norm_labels_multi_mc2 = MinMaxScaler()
    Y_multi_mc2 = norm_labels_multi_mc2.fit_transform(mc_output_multiple_2)

    al_multi_model1 = load_model("Models5/MC_poc/al_multi_model1.h5")
    #al_multi_model1 = generate_multi_network(X_multi, Y_multi_al1)
    #al_multi_model1.save("Models5/MC_poc/al_multi_model1.h5")

    mc_multi_model1 = load_model("Models5/MC_poc/mc_multi_model1.h5")
    #mc_multi_model1 = generate_multi_network(X_multi, Y_multi_mc1)
    #mc_multi_model1.save("Models5/MC_poc/mc_multi_model1.h5")

    al_multi_model2 = load_model("Models5/MC_poc/al_multi_model2.h5")
    #al_multi_model2 = generate_multi_network(X_multi, Y_multi_al2)
    #al_multi_model2.save("Models5/MC_poc/al_multi_model2.h5")

    mc_multi_model2 = load_model("Models5/MC_poc/mc_multi_model2.h5")
    #mc_multi_model2 = generate_multi_network(X_multi, Y_multi_mc2)
    #mc_multi_model2.save("Models5/MC_poc/mc_multi_model2.h5")

    ### Model testing
    spot_plot = np.linspace(start = 75, stop = 125, num = 200)
    spot_plot = np.reshape(spot_plot, (-1, 1))
    eps1_plot = np.reshape(np.repeat(epsilon1, len(spot_plot)), (-1, 1))
    eps2_plot = np.reshape(np.repeat(epsilon2, len(spot_plot)), (-1, 1))
    input_multi_easy = np.concatenate([spot_plot, eps1_plot], axis = 1)
    input_multi_hard = np.concatenate([spot_plot, eps2_plot], axis = 1)
    h = 0.01
    input_good_easy = np.concatenate([
        spot_plot, 
        np.reshape(np.repeat(vol1, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(kappa1, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(theta1, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(epsilon1, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(rho1, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(rate, len(spot_plot)), (-1, 1))
    ], axis = 1)
    input_good_easy_low = np.copy(input_good_easy)
    input_good_easy_low[:, 0] = input_good_easy_low[:, 0] - h
    input_good_hard = np.concatenate([
        spot_plot, 
        np.reshape(np.repeat(vol2, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(kappa2, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(theta2, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(epsilon2, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(rho2, len(spot_plot)), (-1, 1)),
        np.reshape(np.repeat(rate, len(spot_plot)), (-1, 1))
    ], axis = 1) 
    input_good_easy_high = np.copy(input_good_easy)
    input_good_easy_high[:, 0] = input_good_easy_high[:, 0] + h

    test_input = norm_features.transform(spot_plot)
    input_multi_easy = norm_features_multiple.transform(input_multi_easy)
    input_multi_hard = norm_features_multiple.transform(input_multi_hard)

    al_predict1, al_grads1, al_grads1_2 = generate_predictions(test_input, al_model1, norm_features, norm_labels_al1)
    mc_predict1, mc_grads1, mc_grads1_2 = generate_predictions(test_input, mc_model1, norm_features, norm_labels_mc1)
    al_predict2, al_grads2, al_grads2_2 = generate_predictions(test_input, al_model2, norm_features, norm_labels_al2)
    mc_predict2, mc_grads2, mc_grads2_2 = generate_predictions(test_input, mc_model2, norm_features, norm_labels_mc2)

    al_multi_predict1, al_multi_grads1, al_multi_grads1_2 = generate_predictions(input_multi_easy, al_multi_model1, norm_features_multiple, norm_labels_multi_al1)
    mc_multi_predict1, mc_multi_grads1, mc_multi_grads1_2 = generate_predictions(input_multi_easy, mc_multi_model1, norm_features_multiple, norm_labels_multi_mc1)
    al_multi_predict2, al_multi_grads2, al_multi_grads2_2 = generate_predictions(input_multi_hard, al_multi_model2, norm_features_multiple, norm_labels_multi_al2)
    mc_multi_predict2, mc_multi_grads2, mc_multi_grads2_2 = generate_predictions(input_multi_hard, mc_multi_model2, norm_features_multiple, norm_labels_multi_mc2)

    prediction_data = {
        "Andersen Lake" : [al_predict1, al_predict2],
        "Monte Carlo" : [mc_predict1, mc_predict2],
        "Andersen Lake, multi" : [al_multi_predict1, al_multi_predict2],
        "Monte Carlo, multi" : [mc_multi_predict1, mc_multi_predict2]
    }

    delta_data = {
        "Andersen Lake" : [al_grads1, al_grads2],
        "Monte Carlo" : [mc_grads1, mc_grads2],
        "Andersen Lake, multi" : [al_multi_grads1[:,0], al_multi_grads2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1[:,0], mc_multi_grads2[:,0]]
    }

    gamma_data = {
        "Andersen Lake" : [al_grads1_2, al_grads2_2],
        "Monte Carlo" : [mc_grads1_2, mc_grads2_2],
        "Andersen Lake, multi" : [al_multi_grads1_2[:,0], al_multi_grads2_2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1_2[:,0], mc_multi_grads2_2[:,0]]
    }

    plot_func(spot_plot, prediction_data, "MC Predictions")
    plot_func(spot_plot, delta_data, "MC Delta")
    plot_func(spot_plot, gamma_data, "MC Gamma")

    ### Plotting grads for best models
    standardize_5_1000 = "Models5/standardize/standardize_5_1000.h5"
    standardize_5_1000_dict = model_grads(standardize_5_1000, input_good_easy, input_good_hard, some_option)

    standardize_5_100 = "Models5/standardize/standardize_5_100.h5"
    standardize_5_100_dict = model_grads(standardize_5_100, input_good_easy, input_good_hard, some_option)

    standardize_5_500 = "Models5/standardize/standardize_5_500.h5"
    standardize_5_500_dict = model_grads(standardize_5_500, input_good_easy, input_good_hard, some_option)

    standardize_4_50 = "Models5/standardize/standardize_4_50.h5"
    standardize_4_50_dict = model_grads(standardize_4_50, input_good_easy, input_good_hard, some_option)

    tanh_5_50 = "Models5/tanh/tanh_5_50.h5"
    tanh_5_50_dict = model_grads(tanh_5_50, input_good_easy, input_good_hard, some_option)

    tanh_1_50 = "Models5/tanh/tanh_1_50.h5"
    tanh_1_50_dict = model_grads(tanh_1_50, input_good_easy, input_good_hard, some_option)

    tanh_3_50 = "Models5/tanh/tanh_3_50.h5"
    tanh_3_50_dict = model_grads(tanh_3_50, input_good_easy, input_good_hard, some_option)

    ### Mat models
    standardize_mat = "Models5/standardize_mat/standardize_mat_5_1000.h5"
    standardize_mat_dict = model_grads(standardize_mat, input_good_easy, input_good_hard, some_option)

    ### Single models
    standardize_single = "Models5/standardize_single/standardize_single_2_1000.h5"
    standardize_single_dict = model_grads(standardize_single, input_good_easy, input_good_hard, some_option)

    prediction_data = {
        "tanh_3_50" : [tanh_3_50_dict["pred"][0], tanh_3_50_dict["pred"][1]],
        "tanh_1_50" : [tanh_1_50_dict["pred"][0], tanh_1_50_dict["pred"][1]],
        "tanh_5_50" : [tanh_5_50_dict["pred"][0], tanh_5_50_dict["pred"][1]],
        "standardize_5_500" : [standardize_5_500_dict["pred"][0], standardize_5_500_dict["pred"][1]],
        "standardize_5_100" : [standardize_5_100_dict["pred"][0], standardize_5_100_dict["pred"][1]],
        "standardize_5_1000" : [standardize_5_1000_dict["pred"][0], standardize_5_1000_dict["pred"][1]],
        "standardize_4_50" : [standardize_4_50_dict["pred"][0], standardize_4_50_dict["pred"][1]],
        "standardize_mat" : [standardize_mat_dict["pred"][0], standardize_mat_dict["pred"][1]],
        "standardize_single" : [standardize_single_dict["pred"][0], standardize_single_dict["pred"][1]]
    }

    delta_data = {
        "tanh_3_50" : [tanh_3_50_dict["delta"][0], tanh_3_50_dict["delta"][1]],
        "tanh_1_50" : [tanh_1_50_dict["delta"][0], tanh_1_50_dict["delta"][1]],
        "tanh_5_50" : [tanh_5_50_dict["delta"][0], tanh_5_50_dict["delta"][1]],
        "standardize_5_500" : [standardize_5_500_dict["delta"][0], standardize_5_500_dict["delta"][1]],
        "standardize_5_100" : [standardize_5_100_dict["delta"][0], standardize_5_100_dict["delta"][1]],
        "standardize_5_1000" : [standardize_5_1000_dict["delta"][0], standardize_5_1000_dict["delta"][1]],
        "standardize_4_50" : [standardize_4_50_dict["delta"][0], standardize_4_50_dict["delta"][1]],
        "standardize_mat" : [standardize_mat_dict["delta"][0], standardize_mat_dict["delta"][1]],
        "standardize_single" : [standardize_single_dict["delta"][0], standardize_single_dict["delta"][1]]
    }

    gamma_data = {
        "tanh_3_50" : [tanh_3_50_dict["gamma"][0], tanh_3_50_dict["gamma"][1]],
        "tanh_1_50" : [tanh_1_50_dict["gamma"][0], tanh_1_50_dict["gamma"][1]],
        "tanh_5_50" : [tanh_5_50_dict["gamma"][0], tanh_5_50_dict["gamma"][1]],
        "standardize_5_500" : [standardize_5_500_dict["gamma"][0], standardize_5_500_dict["gamma"][1]],
        "standardize_5_100" : [standardize_5_100_dict["gamma"][0], standardize_5_100_dict["gamma"][1]],
        "standardize_5_1000" : [standardize_5_1000_dict["gamma"][0], standardize_5_1000_dict["gamma"][1]],
        "standardize_4_50" : [standardize_4_50_dict["gamma"][0], standardize_4_50_dict["gamma"][1]],
        "standardize_mat" : [standardize_mat_dict["gamma"][0], standardize_mat_dict["gamma"][1]],
        "standardize_single" : [standardize_single_dict["gamma"][0], standardize_single_dict["gamma"][1]]
    }

    plot_func(spot_plot, prediction_data, "Predictions implied volatility models")
    plot_func(spot_plot, delta_data, "Delta implied volatility models")
    plot_func(spot_plot, gamma_data, "Gamma implied volatility models")

    ### Price models
    price_output_normalize_2_50 = "Models5/price_output_normalize/price_output_normalize_2_50.h5"
    price_output_normalize_2_50_dict = model_grads(price_output_normalize_2_50, input_good_easy, input_good_hard, some_option)

    price_output_normalize_3_100 = "Models5/price_output_normalize/price_output_normalize_3_100.h5"
    price_output_normalize_3_100_dict = model_grads(price_output_normalize_3_100, input_good_easy, input_good_hard, some_option)

    price_output_normalize_4_50 = "Models5/price_output_normalize/price_output_normalize_4_50.h5"
    price_output_normalize_4_50_dict = model_grads(price_output_normalize_4_50, input_good_easy, input_good_hard, some_option)

    price_output_normalize_5_500 = "Models5/price_output_normalize/price_output_normalize_5_500.h5"
    price_output_normalize_5_500_dict = model_grads(price_output_normalize_5_500, input_good_easy, input_good_hard, some_option)

    price_output_normalize_5_1000 = "Models5/price_output_normalize/price_output_normalize_5_1000.h5"
    price_output_normalize_5_1000_dict = model_grads(price_output_normalize_5_1000, input_good_easy, input_good_hard, some_option)

    prediction_data_price = {
        "price_output_normalize_2_50" : [price_output_normalize_2_50_dict["pred"][0], price_output_normalize_2_50_dict["pred"][1]],
        "price_output_normalize_3_100" : [price_output_normalize_3_100_dict["pred"][0], price_output_normalize_3_100_dict["pred"][1]],
        "price_output_normalize_4_50" : [price_output_normalize_4_50_dict["pred"][0], price_output_normalize_4_50_dict["pred"][1]],
        "price_output_normalize_5_500" : [price_output_normalize_5_500_dict["pred"][0], price_output_normalize_5_500_dict["pred"][1]],
        "price_output_normalize_5_1000" : [price_output_normalize_5_1000_dict["pred"][0], price_output_normalize_5_1000_dict["pred"][1]]
    }

    delta_data_price = {
        "price_output_normalize_2_50" : [price_output_normalize_2_50_dict["delta"][0], price_output_normalize_2_50_dict["delta"][1]],
        "price_output_normalize_3_100" : [price_output_normalize_3_100_dict["delta"][0], price_output_normalize_3_100_dict["delta"][1]],
        "price_output_normalize_4_50" : [price_output_normalize_4_50_dict["delta"][0], price_output_normalize_4_50_dict["delta"][1]],
        "price_output_normalize_5_500" : [price_output_normalize_5_500_dict["delta"][0], price_output_normalize_5_500_dict["delta"][1]],
        "price_output_normalize_5_1000" : [price_output_normalize_5_1000_dict["delta"][0], price_output_normalize_5_1000_dict["delta"][1]]
    }

    gamma_data_price = {
        "price_output_normalize_2_50" : [price_output_normalize_2_50_dict["gamma"][0], price_output_normalize_2_50_dict["gamma"][1]],
        "price_output_normalize_3_100" : [price_output_normalize_3_100_dict["gamma"][0], price_output_normalize_3_100_dict["gamma"][1]],
        "price_output_normalize_4_50" : [price_output_normalize_4_50_dict["gamma"][0], price_output_normalize_4_50_dict["gamma"][1]],
        "price_output_normalize_5_500" : [price_output_normalize_5_500_dict["gamma"][0], price_output_normalize_5_500_dict["gamma"][1]],
        "price_output_normalize_5_1000" : [price_output_normalize_5_1000_dict["gamma"][0], price_output_normalize_5_1000_dict["gamma"][1]]
    }

    plot_func(spot_plot, prediction_data_price, "Predictions price models")
    plot_func(spot_plot, delta_data_price, "Delta price models")
    plot_func(spot_plot, gamma_data_price, "Gamma price models")

    ### Timing
    timing_list = [standardize_5_1000] + [standardize_5_100] + [standardize_5_500] + [standardize_4_50] + \
        [tanh_5_50] + [tanh_1_50] + [tanh_3_50] + [price_output_normalize_2_50] + [price_output_normalize_3_100] + \
        [price_output_normalize_4_50] + [price_output_normalize_5_500] + [price_output_normalize_5_1000]
    
    timing_results = []
    for some_model in timing_list:
        some_time = timing(some_model, input_good_easy, input_good_hard)
        name = some_model[some_model.rfind("/")+1:]
        timing_results.append([name, some_time])

    al_start = time.time()
    al.Andersen_Lake(model_class_easy, some_option)
    al_time = time.time() - al_start

    ### Monte Carlo
    mc_100_price_5_500 = "Models5/mc_100_price/mc_100_price_5_500.h5"
    mc_100_price_5_500_dict = model_grads(mc_100_price_5_500, input_good_easy, input_good_hard, some_option)

    mc_10_price_4_500 = "Models5/mc_10_price/mc_10_price_4_500.h5"
    mc_10_price_4_500_dict = model_grads(mc_10_price_4_500, input_good_easy, input_good_hard, some_option)

    mc_10000_price_5_100 = "Models5/mc_10000_price/mc_10000_price_5_100.h5"
    mc_10000_price_5_100_dict = model_grads(mc_10000_price_5_100, input_good_easy, input_good_hard, some_option)

    mc_10000_price_4_500 = "Models5/mc_10000_price/mc_10000_price_4_500.h5"
    mc_10000_price_4_500_dict = model_grads(mc_10000_price_4_500, input_good_easy, input_good_hard, some_option)

    mc_1_price_4_100 = "Models5/mc_1_price/mc_1_price_4_100.h5"
    mc_1_price_4_100_dict = model_grads(mc_1_price_4_100, input_good_easy, input_good_hard, some_option)

    mc_10000_5_1000 = "Models5/mc_10000/mc_10000_5_1000.h5"
    mc_10000_5_1000_dict = model_grads(mc_10000_5_1000, input_good_easy, input_good_hard, some_option)

    mc_1000_5_100 = "Models5/mc_1000/mc_1000_5_100.h5"
    mc_1000_5_100_dict = model_grads(mc_1000_5_100, input_good_easy, input_good_hard, some_option)

    mc_100_4_1000 = "Models5/mc_100/mc_100_4_1000.h5"
    mc_100_4_1000_dict = model_grads(mc_100_4_1000, input_good_easy, input_good_hard, some_option)

    mc_prediction_data = {
        "mc_100_price_5_500" : [mc_100_price_5_500_dict["pred"][0], mc_100_price_5_500_dict["pred"][1]],
        "mc_10_price_4_500" : [mc_10_price_4_500_dict["pred"][0], mc_10_price_4_500_dict["pred"][1]],
        "mc_10000_price_5_100" : [mc_10000_price_5_100_dict["pred"][0], mc_10000_price_5_100_dict["pred"][1]],
        "mc_10000_price_4_500" : [mc_10000_price_4_500_dict["pred"][0], mc_10000_price_4_500_dict["pred"][1]],
        "mc_1_price_4_100" : [mc_1_price_4_100_dict["pred"][0], mc_1_price_4_100_dict["pred"][1]],
        "mc_10000_5_1000" : [mc_10000_5_1000_dict["pred"][0], mc_10000_5_1000_dict["pred"][1]],
        "mc_1000_5_100" : [mc_1000_5_100_dict["pred"][0], mc_1000_5_100_dict["pred"][1]],
        "mc_100_4_1000" : [mc_100_4_1000_dict["pred"][0], mc_100_4_1000_dict["pred"][1]]
    }

    mc_delta_data= {
        "mc_100_price_5_500" : [mc_100_price_5_500_dict["delta"][0], mc_100_price_5_500_dict["delta"][1]],
        "mc_10_price_4_500" : [mc_10_price_4_500_dict["delta"][0], mc_10_price_4_500_dict["delta"][1]],
        "mc_10000_price_5_100" : [mc_10000_price_5_100_dict["delta"][0], mc_10000_price_5_100_dict["delta"][1]],
        "mc_10000_price_4_500" : [mc_10000_price_4_500_dict["delta"][0], mc_10000_price_4_500_dict["delta"][1]],
        "mc_1_price_4_100" : [mc_1_price_4_100_dict["delta"][0], mc_1_price_4_100_dict["delta"][1]],
        "mc_10000_5_1000" : [mc_10000_5_1000_dict["delta"][0], mc_10000_5_1000_dict["delta"][1]],
        "mc_1000_5_100" : [mc_1000_5_100_dict["delta"][0], mc_1000_5_100_dict["delta"][1]],
        "mc_100_4_1000" : [mc_100_4_1000_dict["delta"][0], mc_100_4_1000_dict["delta"][1]]
    }

    mc_gamma_data = {
        "mc_100_price_5_500" : [mc_100_price_5_500_dict["gamma"][0], mc_100_price_5_500_dict["gamma"][1]],
        "mc_10_price_4_500" : [mc_10_price_4_500_dict["gamma"][0], mc_10_price_4_500_dict["gamma"][1]],
        "mc_10000_price_5_100" : [mc_10000_price_5_100_dict["gamma"][0], mc_10000_price_5_100_dict["gamma"][1]],
        "mc_10000_price_4_500" : [mc_10000_price_4_500_dict["gamma"][0], mc_10000_price_4_500_dict["gamma"][1]],
        "mc_1_price_4_100" : [mc_1_price_4_100_dict["gamma"][0], mc_1_price_4_100_dict["gamma"][1]],
        "mc_10000_5_1000" : [mc_10000_5_1000_dict["gamma"][0], mc_10000_5_1000_dict["gamma"][1]],
        "mc_1000_5_100" : [mc_1000_5_100_dict["gamma"][0], mc_1000_5_100_dict["gamma"][1]],
        "mc_100_4_1000" : [mc_100_4_1000_dict["gamma"][0], mc_100_4_1000_dict["gamma"][1]]
    }

    plot_func(spot_plot, mc_prediction_data, "Monte Carlo Predictions")
    plot_func(spot_plot, mc_delta_data, "Monte Carlo Delta")
    plot_func(spot_plot, mc_gamma_data, "Monte Carlo Gamma")

    ### Noise
    noise_model = "Models4/noise/noise_5_500.h5"
    noise_dict = model_grads(noise_model, input_good_easy, input_good_hard, some_option, True)

    #mc_price_model = "Models4/mc_10000/price/mc_10000_price_4_100.h5"
    #mc_price_dict = model_grads(mc_price_model, input_good_easy, input_good_hard, some_option, True)

    mc_imp_model = "Models4/mc_10000_include/mc_10000_2_50.h5"
    mc_imp_dict = model_grads(mc_imp_model, input_good_easy, input_good_hard, some_option, True)

    final2_model = "Models4/final2/final2_5_1000.h5"
    final2_dict = model_grads(final2_model, input_good_easy, input_good_hard, some_option, True)

    mix_model = "Models4/activation_functions/mix_5_1000.h5"
    mix_dict = model_grads(mix_model, input_good_easy, input_good_hard, some_option, False)

    new_data_model = "Models4/new_data/new_data_4_500.h5"
    new_data_model_dict = model_grads(new_data_model, input_good_easy, input_good_hard, some_option, False)

    new_data_include = "Models4/new_data_include_zero/include_zero_3_1000.h5"
    new_data_include_dict = model_grads(new_data_include, input_good_easy, input_good_hard, some_option, False)

    prediction_data = {
        "Andersen Lake" : [al_predict1, al_predict2],
        "Monte Carlo" : [mc_predict1, mc_predict2],
        "Andersen Lake, multi" : [al_multi_predict1, al_multi_predict2],
        "Monte Carlo, multi" : [mc_multi_predict1, mc_multi_predict2],
        "Noise model" : [noise_dict["pred"][0], noise_dict["pred"][1]],
        #"MC Price" : [mc_price_dict["pred"][0], mc_price_dict["pred"][1]],
        "MC Imp" : [mc_imp_dict["pred"][0], mc_imp_dict["pred"][1]],
        "Final model" : [final2_dict["pred"][0], final2_dict["pred"][1]],
        "Mix model" : [mix_dict["pred"][0], mix_dict["pred"][1]],
        "New data" : [new_data_model_dict["pred"][0], new_data_model_dict["pred"][1]],
        "New data, include" : [new_data_include_dict["pred"][0], new_data_include_dict["pred"][1]]
    }

    delta_data = {
        "Andersen Lake" : [al_grads1, al_grads2],
        "Monte Carlo" : [mc_grads1, mc_grads2],
        "Andersen Lake, multi" : [al_multi_grads1[:,0], al_multi_grads2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1[:,0], mc_multi_grads2[:,0]],
        "Noise model" : [noise_dict["delta"][0], noise_dict["delta"][1]],
        #"MC Price" : [mc_price_dict["delta"][0], mc_price_dict["delta"][1]],
        "MC Imp" : [mc_imp_dict["delta"][0], mc_imp_dict["delta"][1]],
        "Final model" : [final2_dict["delta"][0], final2_dict["delta"][1]],
        "Mix model" : [mix_dict["delta"][0], mix_dict["delta"][1]],
        "New data" : [new_data_model_dict["delta"][0], new_data_model_dict["delta"][1]],
        "New data, include" : [new_data_include_dict["delta"][0], new_data_include_dict["delta"][1]]
    }

    gamma_data = {
        "Andersen Lake" : [al_grads1_2, al_grads2_2],
        "Monte Carlo" : [mc_grads1_2, mc_grads2_2],
        "Andersen Lake, multi" : [al_multi_grads1_2[:,0], al_multi_grads2_2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1_2[:,0], mc_multi_grads2_2[:,0]],
        "Noise model" : [noise_dict["gamma"][0], noise_dict["gamma"][1]],
        #"MC Price" : [mc_price_dict["gamma"][0], mc_price_dict["gamma"][1]],
        "MC Imp" : [mc_imp_dict["gamma"][0], mc_imp_dict["gamma"][1]],
        "Final model" : [final2_dict["gamma"][0], final2_dict["gamma"][1]],
        "Mix model" : [mix_dict["gamma"][0], mix_dict["gamma"][1]],
        "New data" : [new_data_model_dict["gamma"][0], new_data_model_dict["gamma"][1]],
        "New data, include" : [new_data_include_dict["gamma"][0], new_data_include_dict["gamma"][1]]
    }

    plot_func(spot_plot, prediction_data, "Predictions")
    plot_func(spot_plot, delta_data, "Delta")
    plot_func(spot_plot, gamma_data, "Gamma")

    ### MC
    mc_10000_include = "Models4/mc_10000_include/mc_10000_2_50.h5"

    mc_1000_include = "Models4/mc_1000_include/mc_1000_5_500.h5"

    mc_100_include = "Models4/mc_100_include/mc_100_5_500.h5"

    mc_10_include = "Models4/mc_10_include/mc_10_3_1000.h5"

    mc_1_include = "Models4/mc_1_include/mc_1_4_1000.h5"

    mc_10000 = "Models4/mc_10000/mc_10000_4_1000.h5"

    mc_1000 = "Models4/mc_1000/mc_1000_4_1000.h5"

    mc_100 = "Models4/mc_100/mc_100_4_50.h5"

    mc_10 = "Models4/mc_10/mc_10_1_50.h5"

    ### Different MC models
    # 1 path
    mc_list= [
        #"Models4/mc_1/price/mc_1_price_2_100.h5",
        #"Models4/mc_10/price/mc_10_price_3_50.h5",
        #"Models4/mc_100/price/mc_100_price_5_500.h5",
        #"Models4/mc_1000/price/mc_1000_price_4_100.h5",
        #"Models4/mc_10000/price/mc_10000_price_4_100.h5",
        "Models4/mc_10000/mc_10000_4_1000.h5",
        "Models4/mc_1000/mc_1000_4_1000.h5",
        "Models4/mc_100/mc_100_4_50.h5",
        #"Models4/mc_10/mc_10_1_50.h5",
        "Models4/mc_10000_include/mc_10000_2_50.h5",
        "Models4/mc_1000_include/mc_1000_5_500.h5",
        "Models4/mc_100_include/mc_100_5_500.h5",
        "Models4/mc_10_include/mc_10_3_1000.h5",
        "Models4/mc_1_include/mc_1_4_1000.h5"
    ]
    
    mc_pred = {}
    mc_delta = {}
    mc_gamma = {}
    for model in mc_list:
        name = model[model.find("/")+1:]
        some_dict = model_grads(model, input_good_easy, input_good_hard, some_option, False)
        mc_pred[name] = [some_dict["pred"][0], some_dict["pred"][1]]
        mc_delta[name] = [some_dict["delta"][0], some_dict["delta"][1]]
        mc_gamma[name] = [some_dict["gamma"][0], some_dict["gamma"][1]]

    plot_func(spot_plot, mc_pred, "MC Predictions")
    plot_func(spot_plot, mc_delta, "MC Delta")
    plot_func(spot_plot, mc_gamma, "MC Gamma")

"""
    h = 0.01
    price_easy_al = np.zeros(200)
    price_hard_al = np.zeros(200)
    delta_easy_al = np.zeros(200)
    delta_hard_al = np.zeros(200)
    gamma_easy_al = np.zeros(200)
    gamma_hard_al = np.zeros(200)
    for i in range(len(spot_plot)):
        h = 0.1
        some_spot = spot_plot[i]
        some_spot_low = some_spot - h
        some_spot_low2 = some_spot - 0.5 * h
        some_spot_high = some_spot + h
        some_spot_high2 = some_spot + 0.5 * h

        ### Andersen Lake FD
        ### Easy
        hm_c = hm.HestonClass(some_spot, vol1, kappa1, theta1, epsilon1, rho1, rate)
        hm_low = hm.HestonClass(some_spot_low, vol1, kappa1, theta1, epsilon1, rho1, rate)
        hm_low2 = hm.HestonClass(some_spot_low2, vol1, kappa1, theta1, epsilon1, rho1, rate)

        hm_high = hm.HestonClass(some_spot_high, vol1, kappa1, theta1, epsilon1, rho1, rate)
        hm_high2 = hm.HestonClass(some_spot_high2, vol1, kappa1, theta1, epsilon1, rho1, rate)


        al_f_x = al.Andersen_Lake(hm_c, some_option)
        al_f_x_p = al.Andersen_Lake(hm_high, some_option)
        al_f_x_p2 = al.Andersen_Lake(hm_high2, some_option)
        al_f_x_m = al.Andersen_Lake(hm_low, some_option)
        al_f_x_m2 = al.Andersen_Lake(hm_low2, some_option)

        price_easy_al[i] = al_f_x
        delta_easy_al[i] = ((al_f_x_p2 - al_f_x_m2) / h ) 
        gamma_easy_al[i] = (al_f_x_p - 2 * al_f_x + al_f_x_m) / (h * h)

        ### Hard
        hm_c_hard = hm.HestonClass(some_spot, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_low_hard = hm.HestonClass(some_spot_low, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_high_hard = hm.HestonClass(some_spot_high, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_low_hard2 = hm.HestonClass(some_spot_low2, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_high_hard2 = hm.HestonClass(some_spot_high2, vol2, kappa2, theta2, epsilon2, rho2, rate)

        al_f_x_hard = al.Andersen_Lake(hm_c_hard, some_option)
        al_f_x_p_hard = al.Andersen_Lake(hm_high_hard, some_option)
        al_f_x_m_hard = al.Andersen_Lake(hm_low_hard, some_option)
        al_f_x_p_hard2 = al.Andersen_Lake(hm_high_hard2, some_option)
        al_f_x_m_hard2 = al.Andersen_Lake(hm_low_hard2, some_option)

        price_hard_al[i] = al_f_x_hard
        delta_hard_al[i] = ((al_f_x_p_hard2 - al_f_x_m_hard2) / h ) 
        gamma_hard_al[i] = (al_f_x_p_hard - 2 * al_f_x_hard + al_f_x_m_hard) / (h * h)

    prediction_data["Andersen Lake"] = [price_easy_al, price_hard_al]
    prediction_data_price["Andersen Lake"] = [price_easy_al, price_hard_al]
    plot_func(spot_plot, prediction_data, "Predictions2")
    plot_func(spot_plot, delta_data, "Delta2")
    plot_func(spot_plot, gamma_data, "Gamma2")
    plot_func(spot_plot, prediction_data_price, "Predictions2 price")
    plot_func(spot_plot, delta_data_price, "Delta2 price")
    plot_func(spot_plot, gamma_data_price, "Gamma2 price")

    norm_folder = "Models4/norms/"
    norm_feature_good = joblib.load(norm_folder+"norm_feature.pkl")
    model = load_model("Models4/activation_functions/mix_5_1000.h5")

    model_grads("Models4/activation_functions/mix_5_1000.h5", input_good_easy, input_good_hard, some_option)

    a = norm_feature_good.data_min_[0]
    b = norm_feature_good.data_max_[0]

    inp_tensor_easy = tf.convert_to_tensor(norm_feature_good.transform(input_good_easy))
    inp_tensor_hard = tf.convert_to_tensor(norm_feature_good.transform(input_good_hard))

    ### Andersen Lake model
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_easy)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_easy)
            predict_easy = model(inp_tensor_easy)[:,12]
        grads_easy = tape2.gradient(predict_easy, inp_tensor_easy)[:,0]
    
    grads2_easy = tape.gradient(grads_easy, inp_tensor_easy).numpy()
    grads2_easy = grads2_easy[:,0] / ((b - a) ** 2)
    grads_easy = grads_easy.numpy() / (b - a)

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_hard)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_hard)
            predict_hard = model(inp_tensor_hard)[:,12]
        grads_hard = tape2.gradient(predict_hard, inp_tensor_hard)[:,0]
    
    grads2_hard = tape.gradient(grads_hard, inp_tensor_hard).numpy()
    grads2_hard = grads2_hard[:,0] / ((b - a) ** 2)
    grads_hard = grads_hard.numpy() / (b - a)

    delta_easy = np.zeros(200)
    price_easy = np.zeros(200)
    gamma_easy = np.zeros(200)
    delta_hard = np.zeros(200)
    price_hard = np.zeros(200)
    gamma_hard = np.zeros(200)

    for i in range(200):
        model_bs_easy = bs.BlackScholesForward(spot_plot[i], predict_easy[i], rate)
        model_bs_hard = bs.BlackScholesForward(spot_plot[i], predict_hard[i], rate)
        price_easy[i] = model_bs_easy.BSFormula(some_option)
        delta_easy[i] = model_bs_easy.delta_grads(some_option, grads_easy[i])
        gamma_easy[i] = model_bs_easy.gamma_grads(some_option, grads_easy[i], grads2_easy[i])
        
        price_hard[i] = model_bs_hard.BSFormula(some_option)
        delta_hard[i] = model_bs_hard.delta_grads(some_option, grads_hard[i])
        gamma_hard[i] = model_bs_hard.gamma_grads(some_option, grads_hard[i], grads2_hard[i])



    ### Monte Carlo models
    model_grads("Models4/mc_10000/price/mc_10000_price_4_100.h5", input_good_easy, input_good_hard, some_option)

    plot_mc_dict = {}
    delta_mc_dict = {}
    gamma_mc_dict = {}

    for some_model in mc_1_price:
        name = some_model[some_model.rfind("/")+1:]
        tmp_dict = mc_price_grads(some_model, input_good_easy, input_good_hard)
        plot_mc_dict[name] = tmp_dict["pred"]
        delta_mc_dict[name] = tmp_dict["delta"]
        gamma_mc_dict[name] = tmp_dict["gamma"]

    plot_func(spot_plot, plot_mc_dict, "Predictions MC")
    plot_func(spot_plot, delta_mc_dict, "Delta MC")
    plot_func(spot_plot, gamma_mc_dict, "Gamma MC")

    mc_price_model = load_model("Models4/mc_10000/price/mc_10000_price_4_100.h5")
    some_string = "Models4/mc_10000/price/mc_10000_price_4_100.h5"
    model_dict = mc_price_grads(some_string, input_good_easy, input_good_hard)

    mc_norm_feature = joblib.load("Models4/Heston_input_scale.pkl")
    mc_norm_labels_price = joblib.load("Models4/mc_10000/price/price_scale.pkl")

    inp_tensor_easy_mc = tf.convert_to_tensor(mc_norm_feature.transform(input_good_easy))
    inp_tensor_hard_mc = tf.convert_to_tensor(mc_norm_feature.transform(input_good_hard))

    ### Andersen Lake model
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_easy_mc)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_easy_mc)
            predict_easy_mc = mc_price_model(inp_tensor_easy_mc)[:,12]
        grads_easy_mc = tape2.gradient(predict_easy_mc, inp_tensor_easy_mc)[:,0]
    
    grads2_easy_mc = tape.gradient(grads_easy_mc, inp_tensor_easy_mc).numpy() 
    grads2_easy_mc = grads2_easy_mc[:,0] * np.sqrt(mc_norm_labels_price.var_[12]) / (np.sqrt(mc_norm_feature.var_[0]) ** 2)
    grads_easy_mc = grads_easy_mc.numpy() * np.sqrt(mc_norm_labels_price.var_[12]) / np.sqrt(mc_norm_feature.var_[0])
    price_predictions_easy_mc = mc_norm_labels_price.inverse_transform(mc_price_model(inp_tensor_easy_mc))[:,12]

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_hard_mc)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_hard_mc)
            predict_hard_mc = mc_price_model(inp_tensor_hard_mc)[:,12]
        grads_hard_mc = tape2.gradient(predict_hard_mc, inp_tensor_hard_mc)[:,0]
    
    grads2_hard_mc = tape.gradient(grads_hard_mc, inp_tensor_hard_mc).numpy()
    grads2_hard_mc = grads2_hard_mc[:,0] * np.sqrt(mc_norm_labels_price.var_[12]) / (np.sqrt(mc_norm_feature.var_[0]) ** 2)
    grads_hard_mc = grads_hard_mc.numpy() * np.sqrt(mc_norm_labels_price.var_[12]) / np.sqrt(mc_norm_feature.var_[0])
    price_predictions_hard_mc = mc_norm_labels_price.inverse_transform(mc_price_model(inp_tensor_hard_mc))[:,12]

    prediction_data = {
        "Andersen Lake" : [al_predict1, al_predict2],
        "Monte Carlo" : [mc_predict1, mc_predict2],
        "Andersen Lake, multi" : [al_multi_predict1, al_multi_predict2],
        "Monte Carlo, multi" : [mc_multi_predict1, mc_multi_predict2],
        "Full NN, AL" : [price_easy, price_hard],
        "Full NN, MC" : [price_predictions_easy_mc, price_predictions_hard_mc]
    }

    delta_data = {
        "Andersen Lake" : [al_grads1, al_grads2],
        "Monte Carlo" : [mc_grads1, mc_grads2],
        "Andersen Lake, multi" : [al_multi_grads1[:,0], al_multi_grads2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1[:,0], mc_multi_grads2[:,0]],
        "Neural network" : [delta_easy, delta_hard],
        "Full NN, MC" : [grads_easy_mc, grads_hard_mc]
    }

    gamma_data = {
        "Andersen Lake" : [al_grads1_2, al_grads2_2],
        "Monte Carlo" : [mc_grads1_2, mc_grads2_2],
        "Andersen Lake, multi" : [al_multi_grads1_2[:,0], al_multi_grads2_2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1_2[:,0], mc_multi_grads2_2[:,0]],
        "Neural network" : [gamma_easy, gamma_hard],
        "Full NN, MC" : [grads2_easy_mc, grads2_hard_mc]
    }

    plot_func(spot_plot, prediction_data, "Predictions")
    plot_func(spot_plot, delta_data, "Delta")
    plot_func(spot_plot, gamma_data, "Gamma")




"""