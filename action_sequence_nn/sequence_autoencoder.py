import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Reshape, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 10
MAX_SEQUENCE = 50
BATCH_SIZE = 1
# FEATURES = ['bet', 'call', 'check', 'fold', 'raise']
FEATURES = ['sequence']

def get_train_test_data():
    data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1, names=FEATURES).as_matrix()
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    return train_data, test_data

class KerasBatchGenerator(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.currentIdx = 0

    # First, just return batch size 1
    def generate(self):
        while True:
            # print(self.data[self.currentIdx : self.currentIdx+self.batch_size][0])
            sequences = []
            maxLength = -1
            if self.currentIdx + self.batch_size >= len(self.data):
                self.currentIdx = 0
            # Append all the sequences of this batch into one list
            for i in range(self.currentIdx, self.currentIdx + self.batch_size):
                sequence = ast.literal_eval(self.data[i][0])
                if len(sequence) > maxLength:
                    maxLength = len(sequence)
                # print(sequence)
                sequences.append(sequence)
            # print('before addition: ' + str(sequences))
            # Now pad everything which is not the longest sequence
            for seq in sequences:
                # print('len: ' + str(len(seq)) + ' max len: ' + str(maxLength))
                # print(seq)
                for i in range(len(seq), maxLength):
                    seq.append([0, 0, 0, 0, 0])
            sequencesArr = np.array([np.array(i) for i in sequences])
            # print('after addition: ' + str(sequences))
            # print(sequencesArr.shape)

            # we get sequence with shape (batch_size=1, seqLength, features)
            # sequence = np.expand_dims(sequence, axis=0)
            # sequenceOutput = np.array(sequence)
            # startIndex = sequence.shape[1]
            # if startIndex > MAX_SEQUENCE:
            #     print('ERROR: sequence is bigger than max')
            # paddedSequences = MAX_SEQUENCE - startIndex
            # for i in range(0, paddedSequences):
            #     sequenceOutput = np.insert(sequenceOutput, startIndex + i, np.zeros(5), axis=1)
            # print(self.currentIdx)
            self.currentIdx += self.batch_size
            yield sequencesArr, sequencesArr


train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
train_data_generator = KerasBatchGenerator(train_data, BATCH_SIZE)
test_data_generator = KerasBatchGenerator(test_data, BATCH_SIZE)

input_dim = 5
encoding_dim1 = 1
encoding_dim2 = 3

# Model creation
autoencoder = Sequential()
# With 1 extracted out of 5 features and 1 batch size, we get near 100% accuracy in 7 epochs
# With 1 batch size, it converges quicker
# Encoder
# autoencoder.add(LSTM(input_dim, input_shape=(None, input_dim), return_sequences=True))
autoencoder.add(LSTM(encoding_dim1, input_shape=(None, input_dim), return_sequences=True))
# autoencoder.add(LSTM(encoding_dim2, input_shape=(None, encoding_dim1), return_sequences=True))
# autoencoder.add(Reshape((1, encoding_dim1)))
# LSTM decoder is better than FFN
# autoencoder.add(LSTM(encoding_dim1, input_shape=(None, encoding_dim2), return_sequences=True))
autoencoder.add(LSTM(input_dim, activation='sigmoid', input_shape=(None, encoding_dim1), return_sequences=True))
# autoencoder.add(Dense(250, activation="sigmoid"))
# autoencoder.add(Reshape((50, 5)))


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
# NOTE: for some reason the steps_per_epoch need to be 10 less than train_data//BATCH_SIZE. WHY?
history = autoencoder.fit_generator(train_data_generator.generate(), steps_per_epoch=(len(train_data)//BATCH_SIZE) - 10,
                        epochs=NUM_EPOCHS, validation_data=test_data_generator.generate(),
                        validation_steps=(len(test_data)//BATCH_SIZE) - 10, verbose=1,
                        callbacks=[checkpointer]).history
