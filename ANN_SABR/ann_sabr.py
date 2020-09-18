import numpy as np
from ANN_SABR import model_sabr as ms
from multiprocess import Pool
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import schedules, Adam

train_input = np.loadtxt("train_input_sabr.csv", delimiter=",")
train_output = np.loadtxt("train_output_sabr_approx.csv", delimiter=",")
test_input = np.loadtxt("test_input_sabr.csv", delimiter=",")
test_output = np.loadtxt("test_output_sabr_approx.csv", delimiter=",")


pool = Pool(4)
input_set = [
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr1.h5", 1000, 0.9, 0.1, 0, True, True, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr2.h5", 1000, 0.9, 0.1, 0.2, True, True, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr3.h5", 1000, 0.9, 0.1, 0, True, False, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr4.h5", 1000, 0.9, 0.1, 0.2, True, False, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr5.h5", 1000, 0.9, 0.1, 0, False, True, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr6.h5", 1000, 0.9, 0.1, 0.2, False, True, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr7.h5", 1000, 0.9, 0.1, 0, False, False, 15],
    [train_input, train_output, test_input, test_output, 1000, 100, 0.1, "sabr8.h5", 1000, 0.9, 0.1, 0.2, False, False, 15]
]
res = pool.starmap(ms.model_train, input_set)
print(res)

"""
# Number of nodes
nodes = 1000

# Model creation
model = Sequential()

# Layer 1
model.add(Dense(nodes, input_shape=(14,)))
model.add(Activation('softplus')) # Rectified Linear Unit, f(x) = max(x,0)
model.add(Dropout(0.2))

# Layer 2, output
model.add(Dense(10))
model.add(Activation('elu'))

lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=5000,
        decay_rate=0.95
    )

opt = Adam(learning_rate=lr_schedule)

model.compile(
    loss = 'mse', #mean squared error
    optimizer = opt
)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=2)] #stop if cross validation fails to decrease

model.fit(train_input, train_output, epochs=50, batch_size=128, validation_split=0.1, verbose=1, callbacks=callbacks_list)

model.save('sabr_approx_model2.h5')

model.evaluate(test_input, test_output, verbose = 2)
"""