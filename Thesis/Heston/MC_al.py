import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as k
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import sys
import os
sys.path.append(os.getcwd()) # added for calc server support

from Thesis.Heston import MonteCarlo as mc, AndersenLake as al, HestonModel as hm, NNModelGenerator as mg
from Thesis.misc import VanillaOptions as vo
from Thesis import NeuralNetworkGenerator as nng

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

def plot_func(x_axis : np.array, easy_output : list, hard_output : list, title : str):
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    easy_ax = plt.subplot(121)
    hard_ax = plt.subplot(122)
    easy_ax.plot(x_axis, easy_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    easy_ax.plot(x_axis, easy_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    hard_ax.plot(x_axis, hard_output[0], 'r-', alpha=0.5, label="Andersen Lake")
    hard_ax.plot(x_axis, hard_output[1], 'g-', alpha=0.5, label="Monte Carlo")
    handles, labels = easy_ax.get_legend_handles_labels()
    fig.suptitle(title,fontsize=20)
    easy_ax.set_xlabel("Strike")
    hard_ax.set_xlabel("Strike")
    easy_ax.set_title("Easy case")
    hard_ax.set_title("Hard case")
    fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.1)
    fig.legend(handles, labels, loc="lower center", ncol = 2, fontsize=15)
    plt.savefig("al_mc_"+title+".png")
    plt.close()

def generate_network(X, Y):
    # Modelling
    adam = Adam()

    callbacks_list = [
        LearningRateScheduler(lr_schedule, verbose = 0),
    ]
    model = nng.NN_generator(2, 25, 1, 1)

    model.compile(
        loss = 'mean_squared_error', #mean squared error
        optimizer = adam
    )

    model.fit(X, Y, epochs=100, batch_size=10, verbose = 2, callbacks = callbacks_list, validation_split = 0.1)

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
        grads = tape2.gradient(predict, inp_tensor) / (norm.data_max_ - norm.data_min_)

    grads2 = tape.gradient(grads, inp_tensor) / ((norm.data_max_ - norm.data_min_) ** 2)

    return predict, grads, grads2

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

tau = 1
strike = 100

some_option = vo.EUCall(tau, strike)

print("Calculating single data")
### For single input
al_output1 = np.zeros(len(spot))
mc_output1 = np.zeros(len(spot))
al_output2 = np.zeros(len(spot))
mc_output2 = np.zeros(len(spot))
for i in range(len(spot)):
    some_model = hm.HestonClass(spot[i], vol1, kappa1, theta1, epsilon1, rho1, rate) # case 1
    some_model2 = hm.HestonClass(spot[i], vol2, kappa2, theta2, epsilon2, rho2, rate)
    al_output1[i] = al.Andersen_Lake(some_model, some_option)
    mc_output1[i] = mc.Heston_monte_carlo(some_model, some_option, 10000)
    al_output2[i] = al.Andersen_Lake(some_model2, some_option)
    mc_output2[i] = mc.Heston_monte_carlo(some_model2, some_option, 10000)

plot_func(spot, [al_output1, mc_output1], [al_output2, mc_output2], "Training data")

### Reshaping for nn
spot = np.reshape(spot, (-1, 1))
al_output1 = np.reshape(al_output1, (-1, 1))
mc_output1 = np.reshape(mc_output1, (-1, 1))
al_output2 = np.reshape(al_output2, (-1, 1))
mc_output2 = np.reshape(mc_output2, (-1, 1))

np.savetxt("Data/al_output1.csv", al_output1, delimiter=",")
np.savetxt("Data/mc_output1.csv", mc_output1, delimiter=",")
np.savetxt("Data/al_output2.csv", al_output2, delimiter=",")
np.savetxt("Data/mc_output2.csv", mc_output2, delimiter=",")

print("Calculating multi data")
### For multiple input
al_output_multiple_1 = np.zeros(len(input_array))
mc_output_multiple_1 = np.zeros(len(input_array))
al_output_multiple_2 = np.zeros(len(input_array))
mc_output_multiple_2 = np.zeros(len(input_array))

j = 0
for some_input in input_array:
    some_model = hm.HestonClass(some_input[0], vol1, kappa1, theta1, some_input[1], rho1, rate) # case 1
    some_model2 = hm.HestonClass(some_input[0], vol2, kappa2, theta2, some_input[1], rho2, rate)
    al_output_multiple_1[j] = al.Andersen_Lake(some_model, some_option)
    mc_output_multiple_1[j] = mc.Heston_monte_carlo(some_model, some_option, 1000)
    al_output_multiple_2[j] = al.Andersen_Lake(some_model2, some_option)
    mc_output_multiple_2[j] = mc.Heston_monte_carlo(some_model2, some_option, 1000)
    j += 1

np.savetxt("Data/mc_output_multiple_1.csv", mc_output_multiple_1, delimiter=",")
np.savetxt("Data/mc_output_multiple_2.csv", mc_output_multiple_2, delimiter=",")
np.savetxt("Data/al_output_multiple_1.csv", al_output_multiple_1, delimiter=",")
np.savetxt("Data/al_output_multiple_2.csv", al_output_multiple_2, delimiter=",")


"""
easy_index = input_array[:,1] == 0.5
hard_index = input_array[:,1] == 2

plot_func(spot, [al_output_multiple_1[easy_index], mc_output_multiple_1[easy_index]], \
    [al_output_multiple_2[hard_index], mc_output_multiple_2[hard_index]], "Training data, multiple")

### Ready for NN
norm_features = MinMaxScaler() #MinMaxScaler(feature_range = (-1, 1))
norm_features_multiple = MinMaxScaler()

X = norm_features.fit_transform(spot)
X_multi = norm_features_multiple.fit_transform(input_array)

al_model1 = generate_network(X, al_output1)
mc_model1 = generate_network(X, mc_output1)
al_model2 = generate_network(X, al_output2)
mc_model2 = generate_network(X, mc_output2)

al_multi_model1 = generate_network(X_mutli, al_output_multiple_1)
mc_multi_model1 = generate_network(X_mutli, mc_output_multiple_1)
al_multi_model2 = generate_network(X_mutli, al_output_multiple_2)
mc_multi_model2 = generate_network(X_mutli, mc_output_multiple_2)

### Model testing
spot_plot = np.linspace(start = 75, stop = 125, num = 100)
spot_plot = np.reshape(spot_plot, (-1, 1))

test_input = norm_features.transform(spot_plot)

al_predict1, al_grads1, al_grads1_2 = generate_predictions(test_input, al_model1, norm_features)
mc_predict1, mc_grads1, mc_grads1_2 = generate_predictions(test_input, mc_model1, norm_features)
al_predict2, al_grads2, al_grads2_2 = generate_predictions(test_input, al_model2, norm_features)
mc_predict2, mc_grads2, mc_grads2_2 = generate_predictions(test_input, mc_model2, norm_features)

plot_func(spot_plot, [al_predict1, mc_predict1], [al_predict2, mc_predict2], "Predictions")
plot_func(spot_plot, [al_grads1, mc_grads1], [al_grads2, mc_grads2], "Delta")
plot_func(spot_plot, [al_grads1_2, mc_grads1_2], [al_grads2_2, mc_grads2_2], "Gamma")


al_predictions = al_model.predict(test_input)
mc_predictions = mc_model.predict(test_input)

plt.figure(figsize=(10, 10), dpi = 200)
plt.plot(spot_plot, al_predictions, 'r-', alpha=0.5, label="Andersen Lake")
plt.plot(spot_plot, mc_predictions, 'g-', alpha=0.5, label="Monte Carlo")
plt.legend()
plt.title("Predictions")
plt.xlabel("Strike")
plt.savefig("al_mc_predict.png")
plt.close()

### Derivatives
inp_tensor = tf.convert_to_tensor(test_input)

### Andersen Lake model
with tf.GradientTape() as al_tape:
    al_tape.watch(inp_tensor)
    with tf.GradientTape() as al_tape2:
        al_tape2.watch(inp_tensor)
        al_predict = al_model(inp_tensor)
    al_grads = al_tape2.gradient(al_predict, inp_tensor) / (norm_features.data_max_ - norm_features.data_min_)

al_grads2 = al_tape.gradient(al_grads, inp_tensor) / ((norm_features.data_max_ - norm_features.data_min_) ** 2)

### Monte Carlo model
with tf.GradientTape() as mc_tape:
    mc_tape.watch(inp_tensor)
    with tf.GradientTape() as mc_tape2:
        mc_tape2.watch(inp_tensor)
        mc_predict = mc_model(inp_tensor)
    mc_grads = mc_tape2.gradient(mc_predict, inp_tensor) / (norm_features.data_max_ - norm_features.data_min_)
mc_grads2 = mc_tape.gradient(mc_grads, inp_tensor) / ((norm_features.data_max_ - norm_features.data_min_) ** 2)

plt.figure(figsize=(10, 10), dpi = 200)
plt.plot(spot_plot, al_grads, 'r-', alpha=0.5, label="Andersen Lake")
plt.plot(spot_plot, mc_grads, 'g-', alpha=0.5, label="Monte Carlo")
plt.legend()
plt.title("Delta")
plt.xlabel("Strike")
plt.savefig("al_mc_delta.png")
plt.close()

plt.figure(figsize=(10, 10), dpi = 200)
plt.plot(spot_plot, al_grads2, 'r-', alpha=0.5, label="Andersen Lake")
plt.plot(spot_plot, mc_grads2, 'g-', alpha=0.5, label="Monte Carlo")
plt.legend()
plt.title("Gamma")
plt.xlabel("Strike")
plt.savefig("al_mc_gamma.png")
plt.close()


start = time.time()
al.Andersen_Lake(test_model2, some_option)
stop = time.time()
print("AL: ", stop - start)

start = time.time()
mc.Heston_monte_carlo(test_model2, some_option, 1000)
stop = time.time()
print("MC: ", stop - start)
"""