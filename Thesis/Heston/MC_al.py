import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import joblib
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

from Thesis.Heston import MonteCarlo as mc, AndersenLake as al, HestonModel as hm, NNModelGenerator as mg
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

def plot_func(x_axis : np.array, easy_output : list, hard_output : list, title : str):
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    easy_ax = plt.subplot(121)
    hard_ax = plt.subplot(122)
    easy_ax.plot(x_axis, easy_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    easy_ax.plot(x_axis, easy_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    easy_ax.plot(x_axis, easy_output[2], 'y-', alpha=0.5, label="Andersen Lake, multi")
    easy_ax.plot(x_axis, easy_output[3], 'k-', alpha=0.5, label="Monte Carlo, multi")
    easy_ax.plot(x_axis, easy_output[4], 'm-', alpha=0.5, label="Andersen Lake, FD")
    hard_ax.plot(x_axis, hard_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    hard_ax.plot(x_axis, hard_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    hard_ax.plot(x_axis, hard_output[2], 'y-', alpha=0.5, label="Andersen Lake, multi")
    hard_ax.plot(x_axis, hard_output[3], 'k-', alpha=0.5, label="Monte Carlo, multi")
    hard_ax.plot(x_axis, hard_output[4], 'm-', alpha=0.5, label="Andersen Lake, FD")
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

def generate_predictions(test_x, model, norm):
    ### Derivatives
    inp_tensor = tf.convert_to_tensor(test_x)

    ### Andersen Lake model
    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        with tf.GradientTape() as tape2:
            tape2.watch(inp_tensor)
            predict = model(inp_tensor)
        grads = tape2.gradient(predict, inp_tensor)

    grads2 = tape.gradient(grads, inp_tensor) / ((norm.data_max_ - norm.data_min_) ** 2)

    grads = grads / (norm.data_max_ - norm.data_min_)

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

if __name__ == "__main__":
    ### Generating input data
    spot = np.linspace(start = 50, stop = 150, num = 1000)
    vol1 = 0.04
    vol2 = 0.01
    kappa1 = 2
    kappa2 = 0.1
    theta1 = 0.04
    theta2 = 0.01
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

    ### Plotting training data, multi
    plot_func_training(spot, [al_output1, mc_output1, al_output_multiple_1[easy_index], mc_output_multiple_1[easy_index]], \
        [al_output2, mc_output2, al_output_multiple_2[hard_index], mc_output_multiple_2[hard_index]], "Training data")

    ### Ready for NN
    norm_features = MinMaxScaler() #MinMaxScaler(feature_range = (-1, 1))
    norm_features_multiple = MinMaxScaler()

    X = norm_features.fit_transform(spot)
    X_multi = norm_features_multiple.fit_transform(input_array)

    al_model1 = generate_network(X, al_output1)
    mc_model1 = generate_network(X, mc_output1)
    al_model2 = generate_network(X, al_output2)
    mc_model2 = generate_network(X, mc_output2)

    al_multi_model1 = generate_multi_network(X_multi, al_output_multiple_1)
    mc_multi_model1 = generate_multi_network(X_multi, mc_output_multiple_1)
    al_multi_model2 = generate_multi_network(X_multi, al_output_multiple_2)
    mc_multi_model2 = generate_multi_network(X_multi, mc_output_multiple_2)

    ### Model testing
    spot_plot = np.linspace(start = 75, stop = 125, num = 200)
    spot_plot = np.reshape(spot_plot, (-1, 1))
    eps1_plot = np.reshape(np.repeat(epsilon1, len(spot_plot)), (-1, 1))
    eps2_plot = np.reshape(np.repeat(epsilon2, len(spot_plot)), (-1, 1))
    input_multi_easy = np.concatenate([spot_plot, eps1_plot], axis = 1)
    input_multi_hard = np.concatenate([spot_plot, eps2_plot], axis = 1)
    h = 0.1
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

    al_predict1, al_grads1, al_grads1_2 = generate_predictions(test_input, al_model1, norm_features)
    mc_predict1, mc_grads1, mc_grads1_2 = generate_predictions(test_input, mc_model1, norm_features)
    al_predict2, al_grads2, al_grads2_2 = generate_predictions(test_input, al_model2, norm_features)
    mc_predict2, mc_grads2, mc_grads2_2 = generate_predictions(test_input, mc_model2, norm_features)

    al_multi_predict1, al_multi_grads1, al_multi_grads1_2 = generate_predictions(input_multi_easy, al_multi_model1, norm_features_multiple)
    mc_multi_predict1, mc_multi_grads1, mc_multi_grads1_2 = generate_predictions(input_multi_easy, mc_multi_model1, norm_features_multiple)
    al_multi_predict2, al_multi_grads2, al_multi_grads2_2 = generate_predictions(input_multi_hard, al_multi_model2, norm_features_multiple)
    mc_multi_predict2, mc_multi_grads2, mc_multi_grads2_2 = generate_predictions(input_multi_hard, mc_multi_model2, norm_features_multiple)

    ### Testing the thesis model
    #good_model= load_model()
    norm_folder = "Models4/norms/"
    norm_feature_good = joblib.load(norm_folder+"norm_feature.pkl")
    model = load_model("Models4/activation_functions/mix_5_1000.h5")
    imp_vols_easy = model.predict(norm_feature_good.transform(input_good_easy))[:, 12]
    imp_vols_hard = model.predict(norm_feature_good.transform(input_good_hard))[:, 12]

    price_model = load_model("Models4/price_standard/price_standard_4_500.h5")
    price_norm_feature = joblib.load(norm_folder+"norm_feature_price.pkl")
    price_norm_labels = joblib.load(norm_folder+"norm_labels_price.pkl")

    inp_tensor = tf.convert_to_tensor(norm_feature_good.transform(input_good_easy))
    inp_tensor_price = tf.convert_to_tensor(price_norm_feature.transform(input_good_easy))

    ### Andersen Lake model, price
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor_price)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor_price)
            predict_price = price_model(inp_tensor_price)[:,12]
        grads_price = tape2.gradient(predict_price, inp_tensor_price)

    grads2_price = tape.gradient(grads_price, inp_tensor_price) * np.sqrt(price_norm_labels.var_[12]) / ((price_norm_feature.data_max_ - price_norm_feature.data_min_) ** 2)
    grads_price = grads_price * np.sqrt(price_norm_labels.var_[12]) / (price_norm_feature.data_max_ - price_norm_feature.data_min_)
    predict_price = predict_price * np.sqrt(price_norm_labels.var_[12]) + price_norm_labels.mean_[12]

    ### Andersen Lake model
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(inp_tensor)
        with tf.GradientTape(persistent = True) as tape2:
            tape2.watch(inp_tensor)
            predict = model(inp_tensor)[:,12]
        grads = tape2.gradient(predict, inp_tensor)

    delta = np.zeros(200)
    price = np.zeros(200)
    gamma = np.zeros(200)
    a = norm_feature_good.data_min_[0]
    b = norm_feature_good.data_max_[0]

    for i in range(200):
        model_bs = bs.BlackScholesForward(spot_plot[i], predict[i], rate)
        price[i] = model_bs.BSFormula(some_option)
        delta[i] = model_bs.delta2(some_option, a, b)
        gamma[i] = model_bs.gamma2(some_option, a, b)
        #delta[i] = model_bs.BSVega(some_option) * grads[i,0] / (norm_feature_good.data_max_[0] - norm_feature_good.data_min_[0])

    price_easy_al = np.zeros(200)
    price_hard_al = np.zeros(200)
    delta_easy_al = np.zeros(200)
    delta_hard_al = np.zeros(200)
    gamma_easy_al = np.zeros(200)
    gamma_hard_al = np.zeros(200)
    for i in range(len(imp_vols)):
        h = 0.1
        some_spot = spot_plot[i]
        some_spot_low = some_spot - h
        some_spot_high = some_spot + h

        ### Andersen Lake FD
        ### Easy
        hm_c = hm.HestonClass(some_spot, vol1, kappa1, theta1, epsilon1, rho1, rate)
        hm_low = hm.HestonClass(some_spot_low, vol1, kappa1, theta1, epsilon1, rho1, rate)
        hm_high = hm.HestonClass(some_spot_high, vol1, kappa1, theta1, epsilon1, rho1, rate)

        al_f_x = al.Andersen_Lake(hm_c, some_option)
        al_f_x_p = al.Andersen_Lake(hm_high, some_option)
        al_f_x_m = al.Andersen_Lake(hm_low, some_option)

        price_easy_al[i] = al_f_x
        delta_easy_al[i] = ((al_f_x_p - al_f_x) / h ) 
        gamma_easy_al[i] = (al_f_x_p - 2 * al_f_x + al_f_x_m) / (h * h)

        ### Hard
        hm_c_hard = hm.HestonClass(some_spot, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_low_hard = hm.HestonClass(some_spot_low, vol2, kappa2, theta2, epsilon2, rho2, rate)
        hm_high_hard = hm.HestonClass(some_spot_high, vol2, kappa2, theta2, epsilon2, rho2, rate)

        al_f_x_hard = al.Andersen_Lake(hm_c_hard, some_option)
        al_f_x_p_hard = al.Andersen_Lake(hm_high_hard, some_option)
        al_f_x_m_hard = al.Andersen_Lake(hm_low_hard, some_option)

        price_hard_al[i] = al_f_x_hard
        delta_hard_al[i] = ((al_f_x_p_hard - al_f_x_hard) / h ) 
        gamma_hard_al[i] = (al_f_x_p_hard - 2 * al_f_x_hard + al_f_x_m_hard) / (h * h)

    plot_func(spot_plot, [al_predict1, mc_predict1, al_multi_predict1, mc_multi_predict1, price_easy_al], [al_predict2, mc_predict2, al_multi_predict2, mc_multi_predict2, price_hard_al], "Predictions")
    plot_func(spot_plot, [al_grads1, mc_grads1, al_multi_grads1[:,0], mc_multi_grads1[:,0], delta_easy_al], [al_grads2, mc_grads2, al_multi_grads2[:,0], mc_multi_grads2[:,0], delta_hard_al], "Delta")
    plot_func(spot_plot, [al_grads1_2, mc_grads1_2, al_multi_grads1_2[:,0], mc_multi_grads1_2[:,0], gamma_easy_al], [al_grads2_2, mc_grads2_2, al_multi_grads2_2[:,0], mc_multi_grads2_2[:,0], gamma_hard_al], "Gamma")



