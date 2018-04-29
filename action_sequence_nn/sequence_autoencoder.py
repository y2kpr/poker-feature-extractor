import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras.models import Sequential
from keras.layers import LSTM, Reshape, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 10
MAX_SEQUENCE = 50
# FEATURES = ['bet', 'call', 'check', 'fold', 'raise']
FEATURES = ['sequence']

def get_train_test_data():
    data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1, names=FEATURES).as_matrix()
    train_data, test_data = train_test_split(data, test_size=0.2)

    return train_data, test_data

class KerasBatchGenerator(object):

    def __init__(self, data):
        self.data = data
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.currentIdx = 0

    # First, just return batch size 1
    def generate(self):
        # print(self.data[self.currentIdx][0])
        sequence = np.array(ast.literal_eval(self.data[self.currentIdx][0]))
        # we get sequence with shape (batch_size=1, seqLength, features)
        sequence = np.expand_dims(sequence, axis=0)

        sequenceOutput = np.array(sequence)
        startIndex = sequence.shape[1]
        if startIndex > MAX_SEQUENCE:
            print('ERROR: sequence is bigger than max')
        paddedSequences = MAX_SEQUENCE - startIndex
        for i in range(0, paddedSequences):
            sequenceOutput = np.insert(sequenceOutput, startIndex + i, np.zeros(5), axis=1)
        # print('sequence is ' + str(sequence.shape))
        # print('sequence output is ' + str(sequenceOutput.shape))
        # print(sequenceOutput.flatten())
        self.currentIdx += 1
        while True:
            yield sequence, sequenceOutput


train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
train_data_generator = KerasBatchGenerator(train_data)
test_data_generator = KerasBatchGenerator(test_data)

encoding_dim1 = 50

# Model creation
autoencoder = Sequential()
# With 150 extracted features, val_accuracy gets to 88% in one epoch with 1000 sequences
# Encoder
autoencoder.add(LSTM(encoding_dim1, input_shape=(None, 5)))
autoencoder.add(Reshape((1, encoding_dim1)))
# LSTM decoder is better than FFN
autoencoder.add(LSTM(250, activation='sigmoid', input_shape=(1, encoding_dim1)))
# autoencoder.add(Dense(250, activation="sigmoid"))
autoencoder.add(Reshape((50, 5)))


# TODO: learn how categorical_crossentropy and categorical_accuracy work
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy, metrics.categorical_accuracy])

checkpointer = ModelCheckpoint(filepath="models/sequence-model.h5",
                               verbose=0,
                               save_best_only=True)
# tensorboard = TensorBoard(log_dir='./logs',
                          # histogram_freq=0,
                          # write_graph=True,
                          # write_images=True)
# history = autoencoder.fit(train_data, train_data,
#                     epochs=10,
#                     validation_data=(test_data, test_data),
#                     verbose=1,
#                     callbacks=[checkpointer, tensorboard]).history

# In the future, include batch size in steps_per_epoch
# QUESTION: what effect does batch size really have when backprop is not done between batches?
history = autoencoder.fit_generator(train_data_generator.generate(), steps_per_epoch=len(train_data),
                        epochs=NUM_EPOCHS, validation_data=test_data_generator.generate(),
                        validation_steps=len(test_data), verbose=1,
                        callbacks=[checkpointer]).history
