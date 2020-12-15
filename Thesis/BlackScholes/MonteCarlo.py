import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import joblib
import glob
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras import backend as k
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from multiprocessing import Pool, cpu_count
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.misc import VanillaOptions as vo
from Thesis import NeuralNetworkGenerator as nng
from Thesis.BlackScholes import BlackScholes as bs, MC_generator as mc

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

def plot_func(x_axis : np.array, plot_data : dict, title : str):
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    easy_ax = plt.subplot(121)
    hard_ax = plt.subplot(122)
    color=iter(plt.cm.tab10(np.linspace(0,1,len(plot_data))))
    for key in plot_data:
        if key == "Black 76":
            c = 'black'
        else:
            c = next(color)
        easy_ax.plot(x_axis, plot_data[key][0], color = c, label = key, alpha = 0.5, linewidth=2)
        hard_ax.plot(x_axis, plot_data[key][1], color = c, label = key, alpha = 0.5, linewidth=2)
        
    handles, labels = easy_ax.get_legend_handles_labels()
    fig.suptitle(title, fontsize=25)
    easy_ax.tick_params(axis = "both", labelsize=15)
    hard_ax.tick_params(axis = "both", labelsize=15)
    easy_ax.set_xlabel("Spot", fontsize = 20)
    hard_ax.set_xlabel("Spot", fontsize = 20)
    easy_ax.set_title("Low volatility", fontsize = 20)
    hard_ax.set_title("High volatilty", fontsize = 20)
    fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.15)
    fig.legend(handles, labels, loc="lower center", ncol = 5, prop={'size': 20})
    plt.savefig("Final_plots/BS/bs_mc_"+title.replace(" ", "_").replace(",","")+".png")
    plt.close()

def generate_network(X, Y) -> list:
    # Modelling
    adam = Adam()

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
    ]
    model = nng.NN_generator(4, 25, np.shape(X)[1], np.shape(Y)[1])

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    loss_history = model.fit(X, Y, epochs=100, batch_size=128, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)

    return loss_history, model

def generate_multi_network(X, Y):
    # Modelling
    adam = Adam()

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
    ]
    model = nng.NN_generator(4, 25, np.shape(X)[1], np.shape(Y)[1])

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    loss_history = model.fit(X, Y, epochs=100, batch_size=1024, verbose = 0, callbacks = callbacks_list, validation_split = 0.1, shuffle=True)

    return loss_history, model

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

def model_grads(model_string : str, easy_case : np.ndarray, hard_case : np.ndarray, option : vo.VanillaOption) -> dict:
    model = load_model(model_string)
    model_folder = model_string[:model_string.rfind("/") + 1]
    norm_feature = joblib.load(model_folder+"norm_feature.pkl")

    if os.path.exists(model_folder+"/norm_labels.pkl"):
        norm_labels = joblib.load(model_folder+"norm_labels.pkl")
        normal_out = True
    else:
        normal_out = False

    if isinstance(norm_feature, MinMaxScaler):
        if normal_out:
            grads_scale = np.sqrt(norm_labels.var_[12]) / (norm_feature.data_max_[0] - norm_feature.data_min_[0])
            grads2_scale = np.sqrt(norm_labels.var_[12]) / ((norm_feature.data_max_[0] - norm_feature.data_min_[0]) ** 2)
        else:
            grads_scale = 1 / (norm_feature.data_max_[0] - norm_feature.data_min_[0])
            grads2_scale = 1 / ((norm_feature.data_max_[0] - norm_feature.data_min_[0]) ** 2)
    else:
        if normal_out:
            grads_scale = np.sqrt(norm_labels.var_[12]) / np.sqrt(norm_feature.var_[0])
            grads2_scale = np.sqrt(norm_labels.var_[12]) / (np.sqrt(norm_feature.var_[0]) ** 2)
        else:
            grads_scale = 1 / np.sqrt(norm_feature.var_[0])
            grads2_scale = 1 / (np.sqrt(norm_feature.var_[0]) ** 2)

    inp_tensor_easy = tf.convert_to_tensor(norm_feature.transform(easy_case))
    inp_tensor_hard = tf.convert_to_tensor(norm_feature.transform(hard_case))

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_easy)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_easy)
            predict_easy = model(inp_tensor_easy)[:,12]
        grads_easy = tape2.gradient(predict_easy, inp_tensor_easy)[:,0]
    
    grads2_easy = tape.gradient(grads_easy, inp_tensor_easy).numpy() 
    grads2_easy = grads2_easy[:,0] * grads2_scale
    grads_easy = grads_easy.numpy() * grads_scale
    try:
        predict_easy = norm_labels.inverse_transform(model(inp_tensor_easy))[:,12]
    except:
        predict_easy = model(inp_tensor_easy)[:,12]

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_hard)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_hard)
            predict_hard = model(inp_tensor_hard)[:,12]
        grads_hard = tape2.gradient(predict_hard, inp_tensor_hard)[:,0]
    
    grads2_hard = tape.gradient(grads_hard, inp_tensor_hard).numpy()
    grads2_hard = grads2_hard[:,0] * grads2_scale
    grads_hard = grads_hard.numpy() * grads_scale
    try:
        predict_hard = norm_labels.inverse_transform(model(inp_tensor_hard))[:,12]
    except:
        predict_hard = model(inp_tensor_hard)[:,12]

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

if __name__ == "__main__":
    ### Generating input data
    spot = np.linspace(start = 50, stop = 150, num = 5000)
    vol = np.linspace(start = 0.01, stop = 0.5, num = 20)
    vol1 = vol[5]
    vol2 = vol[15]
    rate = 0.05
    input_array = np.array(list(itertools.product(spot, vol)))

    tau = 1 #set to match option data
    strike = 100

    some_option = vo.EUCall(tau, strike)

    spot = np.reshape(spot, (-1, 1))

    ### Single data
    output_low_vol = np.zeros(len(spot))
    output_bs_low = np.zeros(len(spot))
    output_high_vol = np.zeros(len(spot))
    output_bs_high = np.zeros(len(spot))
    i = 0
    single_start = time.time()
    for some_spot in spot:
        some_model1 = bs.BlackScholesForward(some_spot, vol1, rate)
        some_model2 = bs.BlackScholesForward(some_spot, vol2, rate)
        output_low_vol[i] = mc.Black_monte_carlo(some_model1, some_option, 5000)
        output_bs_low[i] = some_model1.BSFormula(some_option)
        output_high_vol[i] = mc.Black_monte_carlo(some_model2, some_option, 5000)
        output_bs_high[i] = some_model2.BSFormula(some_option)
        i += 1
    single_stop = time.time()
    single_stop - single_start

    low_index = input_array[:,1] == vol1
    high_index = input_array[:,1] == vol2
    output_multi = np.zeros(np.shape(input_array)[0])
    output_multi_bs = np.zeros(np.shape(input_array)[0])
    i = 0
    multi_start = time.time()
    for some_spot, some_vol in input_array:
        some_model = bs.BlackScholesForward(some_spot, some_vol, rate)
        output_multi[i] = mc.Black_monte_carlo(some_model, some_option, 5000)
        output_multi_bs[i] = some_model.BSFormula(some_option)
        i += 1
    multi_stop = time.time()
    multi_stop - multi_start
    training_data = {
        "Black" : [output_bs_low, output_bs_high],
        "Monte Carlo" : [output_low_vol, output_high_vol],
        "Black, multi" : [output_multi_bs[low_index], output_multi_bs[high_index]],
        "Monte Carlo, multi" : [output_multi[low_index], output_multi[high_index]]
    }
    
    plot_func(spot, training_data, "Training data")

    output_bs_low = np.reshape(output_bs_low, (-1, 1))
    output_bs_high = np.reshape(output_bs_high, (-1, 1))
    output_low_vol = np.reshape(output_low_vol, (-1, 1))
    output_high_vol = np.reshape(output_high_vol, (-1, 1))
    output_multi_bs = np.reshape(output_multi_bs, (-1, 1))
    output_multi = np.reshape(output_multi, (-1, 1))
    
    ### Ready for NN
    norm_features = StandardScaler()
    norm_features_multiple = StandardScaler()
    X = norm_features.fit_transform(spot)
    X_multi = norm_features_multiple.fit_transform(input_array)

    norm_labels_bs1 = MinMaxScaler()
    Y_bs1 = norm_labels_bs1.fit_transform(output_bs_low)
    norm_labels_mc1 = MinMaxScaler()
    Y_mc1 = norm_labels_mc1.fit_transform(output_low_vol)
    norm_labels_bs2 = MinMaxScaler()
    Y_bs2 = norm_labels_bs2.fit_transform(output_bs_high)
    norm_labels_mc2 = MinMaxScaler()
    Y_mc2 = norm_labels_mc2.fit_transform(output_high_vol)

    #bs_model1 = load_model("Models5/BS/bs_model1.h5")
    single_model_start = time.time()
    loss_bs_model1, bs_model1 = generate_network(X, Y_bs1)
    single_model_stop = time.time()
    bs_model1.save("Models5/BS/bs_model1.h5")
    #mc_model1 = load_model("Models5/BS/mc_model1.h5")
    loss_mc_model1, mc_model1 = generate_network(X, Y_mc1)
    mc_model1.save("Models5/BS/mc_model1.h5")
    #bs_model2 = load_model("Models5/BS/bs_model2.h5")
    loss_bs_model2, bs_model2 = generate_network(X, Y_bs2)
    bs_model2.save("Models5/BS/bs_model2.h5")
    #mc_model2 = load_model("Models5/BS/mc_model2.h5")
    loss_mc_model2, mc_model2 = generate_network(X, Y_mc2)
    mc_model2.save("Models5/BS/mc_model2.h5")

    norm_labels_multi_bs1 = MinMaxScaler()
    Y_multi_bs1 = norm_labels_multi_bs1.fit_transform(output_multi_bs)
    norm_labels_multi_mc1 = MinMaxScaler()
    Y_multi_mc1 = norm_labels_multi_mc1.fit_transform(output_multi)

    multi_model_start = time.time()
    #bs_multi_model1 = load_model("Models5/BS/bs_multi_model1.h5")
    loss_bs_multi_model1, bs_multi_model1 = generate_multi_network(X_multi, Y_multi_bs1)
    multi_model_stop = time.time()
    bs_multi_model1.save("Models5/BS/bs_multi_model1.h5")

    #mc_multi_model1 = load_model("Models5/BS/mc_multi_model1.h5")
    loss_mc_multi_model1, mc_multi_model1 = generate_multi_network(X_multi, Y_multi_mc1)
    mc_multi_model1.save("Models5/BS/mc_multi_model1.h5")

    ### Model testing
    spot_plot = np.linspace(start = 75, stop = 125, num = 200)
    spot_plot = np.reshape(spot_plot, (-1, 1))
    vol1_plot = np.reshape(np.repeat(vol1, len(spot_plot)), (-1, 1))
    vol2_plot = np.reshape(np.repeat(vol2, len(spot_plot)), (-1, 1))
    input_multi_low = np.concatenate([spot_plot, vol1_plot], axis = 1)
    input_multi_high = np.concatenate([spot_plot, vol2_plot], axis = 1)

    test_input = norm_features.transform(spot_plot)
    input_multi_low = norm_features_multiple.transform(input_multi_low)
    input_multi_high = norm_features_multiple.transform(input_multi_high)

    bs_predict1, bs_grads1, bs_grads1_2 = generate_predictions(test_input, bs_model1, norm_features, norm_labels_bs1)
    mc_predict1, mc_grads1, mc_grads1_2 = generate_predictions(test_input, mc_model1, norm_features, norm_labels_mc1)
    bs_predict2, bs_grads2, bs_grads2_2 = generate_predictions(test_input, bs_model2, norm_features, norm_labels_bs2)
    mc_predict2, mc_grads2, mc_grads2_2 = generate_predictions(test_input, mc_model2, norm_features, norm_labels_mc2)

    bs_multi_predict1, bs_multi_grads1, bs_multi_grads1_2 = generate_predictions(input_multi_low, bs_multi_model1, norm_features_multiple, norm_labels_multi_bs1)
    mc_multi_predict1, mc_multi_grads1, mc_multi_grads1_2 = generate_predictions(input_multi_low, mc_multi_model1, norm_features_multiple, norm_labels_multi_mc1)
    bs_multi_predict2, bs_multi_grads2, bs_multi_grads2_2 = generate_predictions(input_multi_high, bs_multi_model1, norm_features_multiple, norm_labels_multi_bs1)
    mc_multi_predict2, mc_multi_grads2, mc_multi_grads2_2 = generate_predictions(input_multi_high, mc_multi_model1, norm_features_multiple, norm_labels_multi_mc1)

    ### Generate benchmark data
    ### Single data
    pred_low = np.zeros(np.shape(spot_plot)[0])
    pred_high = np.zeros(np.shape(spot_plot)[0])
    delta_low = np.zeros(np.shape(spot_plot)[0])
    delta_high = np.zeros(np.shape(spot_plot)[0])
    gamma_low = np.zeros(np.shape(spot_plot)[0])
    gamma_high = np.zeros(np.shape(spot_plot)[0])

    i = 0
    for some_spot in spot_plot:
        some_model1 = bs.BlackScholesForward(some_spot, vol1, rate)
        some_model2 = bs.BlackScholesForward(some_spot, vol2, rate)
        pred_low[i] = some_model1.BSFormula(some_option)
        pred_high[i] = some_model2.BSFormula(some_option)
        delta_low[i] = some_model1.BSDelta(some_option)
        delta_high[i] = some_model2.BSDelta(some_option)
        gamma_low[i] = some_model1.BSGamma(some_option)
        gamma_high[i] = some_model2.BSGamma(some_option)
        i += 1

    prediction_data = {
        "Black, ANN" : [bs_predict1, bs_predict2],
        "Monte Carlo" : [mc_predict1, mc_predict2],
        "Black, multi" : [bs_multi_predict1, bs_multi_predict2],
        "Monte Carlo, multi" : [mc_multi_predict1, mc_multi_predict2],
        "Black 76" : [pred_low, pred_high]
    }

    delta_data = {
        "Black, ANN" : [bs_grads1, bs_grads2],
        "Monte Carlo" : [mc_grads1, mc_grads2],
        "Black, multi" : [bs_multi_grads1[:,0], bs_multi_grads2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1[:,0], mc_multi_grads2[:,0]],
        "Black 76" : [delta_low, delta_high]
    }

    gamma_data = {
        "Black, ANN" : [bs_grads1_2, bs_grads2_2],
        "Monte Carlo" : [mc_grads1_2, mc_grads2_2],
        "Black, multi" : [bs_multi_grads1_2[:,0], bs_multi_grads2_2[:,0]],
        "Monte Carlo, multi" : [mc_multi_grads1_2[:,0], mc_multi_grads2_2[:,0]],
        "Black 76" : [gamma_low, gamma_high]
    }

    plot_func(spot_plot, prediction_data, "MC Predictions")
    plot_func(spot_plot, delta_data, "MC Delta")
    plot_func(spot_plot, gamma_data, "MC Gamma")
