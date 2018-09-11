import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras import regularizers
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Reshape, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
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

    def get_padded_sequence(self, sequence, length):
        assert (length > len(sequence))

        sequenceOutput = np.array(sequence)
        # padding = np.zeros((MAX_SEQUENCE - len(sequencesArr), 5))
        # TODO: figure out cocatenate
        # np.concatenate(sequencesOutputArr, len(sequencesArr), padding)
        for i in range(sequence.shape[1], MAX_SEQUENCE):
            # print('i is ' + str(i))
            sequenceOutput = np.insert(sequenceOutput, i, np.zeros(5), axis=1)
        # print('sequence: ' + str(sequencesArr.shape))
        # print('sequence output: ' + str(sequencesOutputArr.shape))
        return sequenceOutput

    # input and output (which is a padded output) shape: [batch_size, seq_length, features]
    # Returns shape: [batch_size, seq_length, [features, mask]]
    # mask: 1 = real input, 0 = padded input
    def mask_sequences(self, input, output):
        assert (len(input) == len(output))
        x = np.zeros((input.shape[0], input.shape[1], 6))
        y = np.zeros((output.shape[0], output.shape[1], 6))
        print('input shape: ' + str(input.shape))
        print('x shape: ' + str(x.shape))
        for i, (ielem, oelem) in enumerate(zip(input, output)):
            print(ielem.shape)
            # insert 0 at axis=1
            # print(oelem)

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
            self.currentIdx += self.batch_size

            sequencesOutputArr = self.get_padded_sequence(sequencesArr, MAX_SEQUENCE)
            # sequencesArr, sequencesOutputArr =
            # self.mask_sequences(sequencesArr, sequencesOutputArr)

            yield sequencesArr, sequencesOutputArr


def sequence_loss():
    """
    Basically just categorical cross entropy masked for where the mask==2 (output)
    :return:
    """

    def loss(y_true, y_pred):
        # return 0 if max of y_true is 0. Ignores padded sequences
        if K.max(y_true) == 0:
            return 0.
        return K.categorical_crossentropy(y_true, y_pred)

    return loss

def sequence_accuracy(y_true, y_pred):
    """
    A categorical_accuracy function that ignores padded values
    """
    if K.max(y_true) == 0:
        return 1.
    return metrics.categorical_accuracy(y_true, y_pred)

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
train_data_generator = KerasBatchGenerator(train_data, BATCH_SIZE)
test_data_generator = KerasBatchGenerator(test_data, BATCH_SIZE)

input_dim = 5
encoding_dim1 = 1
# encoding_dim2 = 3

# Model creation
autoencoder = Sequential()
# With 1 extracted out of 5 features and 1 batch size, we get near 100% accuracy in 7 epochs
# With 1 batch size, it converges quicker
# Encoder
# autoencoder.add(LSTM(input_dim, input_shape=(None, input_dim), return_sequences=True))
autoencoder.add(LSTM(50, input_shape=(None, input_dim), return_sequences=False))
# autoencoder.add(LSTM(encoding_dim2, input_shape=(None, encoding_dim1), return_sequences=True))
# autoencoder.add(Reshape((1, encoding_dim1)))
# LSTM decoder is better than FFN
# autoencoder.add(LSTM(encoding_dim1, input_shape=(None, encoding_dim2), return_sequences=True))
# autoencoder.add(LSTM(input_dim, activation='sigmoid', input_shape=(None, encoding_dim1), return_sequences=True))
autoencoder.add(Dense(250, activation="sigmoid"))
autoencoder.add(Reshape((50, 5)))


# TODO: learn how categorical_crossentropy and categorical_accuracy work
autoencoder.compile(optimizer='adam', loss=sequence_loss(), metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, sequence_accuracy])

checkpointer = ModelCheckpoint(filepath="models/sequence-model-{epoch:02d}.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
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
                        callbacks=[checkpointer, tensorboard]).history
