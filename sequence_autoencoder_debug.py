import pandas as pd
import numpy as np
import keras.metrics as metrics
from keras.models import Sequential
from keras.layers import LSTM, Reshape
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

FEATURES = ['bet', 'call', 'check', 'fold', 'raise']

def get_train_test_data():
    data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1, names=FEATURES, nrows=80).as_matrix()

    # We're assuming that eight is the average sequence length of a poker game.
    # Fill the 50 time steps with 8 of real data and 42 padded steps of zeroes.
    # TODO: Later, work on doing a normal distribution of sequence length centered at 8.
    data = np.reshape(data, (data.shape[0]//8,8,5))
    padded_data = data
    # i = 8
    # while i <= data.shape[0]:
    #     # very expensive operation because insert creates a new copy each time.
    #     # TODO: figure out how to insert all paddings in one go
    #     padded_data = np.insert(padded_data, i, np.zeros((42,5)), axis=0)
    #     # print(str(i) + ' shape is: ' + str(data.shape[0]))
    #     print(i)
    #     i += 8
    for i in range(0, 42):
        padded_data = np.insert(padded_data, 8 + i, np.zeros(5), axis=1)
    train_data, test_data = train_test_split(padded_data, test_size=0.2)

    return train_data, test_data

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))

# Model creation
autoencoder = Sequential()
# TODO: use sigmoid activation (in the final decoder output) because our inputs are 0 or 1
# Encoder
autoencoder.add(LSTM(150, input_shape=(50, 5)))
# Reshaping is a manual interference that is usually not good for NNs to predict the feautres
autoencoder.add(Reshape((50, 3)))
# decoder
autoencoder.add(LSTM(5, activation='sigmoid', input_shape=(50, 3), return_sequences=True))

# TODO: learn how categorical_crossentropy works
# categorical_accuracy may not be appropriate because of the padded zeros without a max value
# Either I have to createa custom categorical_accuracy that ignores the padded zeros or use
# binary_accuracy
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])

checkpointer = ModelCheckpoint(filepath="models/sequence-model-{epoch:02d}.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.predict(train_data)
np.set_printoptions(threshold=np.inf)
print(history)
print('------------------------------------')
print(train_data)
print(history.shape)
