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
from tensorflow.python import debug as tf_debug

NUM_EPOCHS = 100
# maximum number of actions allowed in a sequence
MAX_SEQUENCE = 50
BATCH_SIZE = 10
INPUT_DIM = 5
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

    '''
    Takes a 2D sequence of any length and pads zero-sequences so that the resulting
    sequence is of length 'length'
    '''
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

    '''
    In a list of game sequences (given as self.data), generate a batch of sequences
    that are padded to the maximum sequence length in that batch.
    Checked that it works accurately
    '''
    def generate(self):
        while True:
            # print(self.data[self.currentIdx : self.currentIdx+self.batch_size][0])
            sequences = []
            maxLength = -1
            # what is this doing?
            if self.currentIdx + self.batch_size >= len(self.data):
                # print('finished one epoch. Setting idx back to 0')
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

            # Since output is always padded to MAX_SEQUENCE, the
            sequencesOutputArr = self.get_padded_sequence(sequencesArr, MAX_SEQUENCE)
            # sequencesArr, sequencesOutputArr =
            # self.mask_sequences(sequencesArr, sequencesOutputArr)
            # print('sequence arr shape: ' + str(sequencesArr.shape))
            # print('sequence output arr shape: ' + str(sequencesOutputArr.shape))

            yield sequencesArr, sequencesOutputArr


def sequence_loss():
    """
    Basically just categorical cross entropy masked to exclude the padded sequences
    :return:
    """

    def loss(y_true, y_pred):
        # return 0 if max of y_true is 0. Ignores padded sequences
        # if K.max(y_true) == 0:
        #     return 0.
        return K.categorical_crossentropy(y_true, y_pred)

    return loss

# QUESTION: does y_true have one game sequence or one action in a game sequence?
def sequence_accuracy(y_true, y_pred):
    """
    A categorical_accuracy function that ignores padded values
    y_true shape = y_pred shape = batch_size x time steps x input dim (verified)

    needs to be accurate ONLY if all actions in a game (sequence) are predicted correctly
    """

    # setting size to number of games in y_true
    shape = K.int_shape(y_true)
    accuracy = np.ones(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        for j in range(MAX_SEQUENCE):
            # TODO: figure out how to make this logic work with tensors
            # K.switch(K.max(y_true[i][j]) >= K.variable(value=0)
            #   and K.argmax(y_true[i][j]) != K.argmax(y_pred[i][j])
            if K.argmax(y_true[i][j]) != K.argmax(y_pred[i][j]):
                # if K.argmax(y_true[i][j]) != K.argmax(y_pred[i][j]):
                accuracy[i] = 0
                break

    return K.cast(accuracy, K.floatx())

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
train_data_generator = KerasBatchGenerator(train_data, BATCH_SIZE)
test_data_generator = KerasBatchGenerator(test_data, BATCH_SIZE)


# Model creation
autoencoder = Sequential()
# Encoder
autoencoder.add(LSTM(10, input_shape=(None, INPUT_DIM), return_sequences=False))
# Decoder
# LSTM decoder gets better accuracy than FFN but it does not consider multiple time steps at once
autoencoder.add(Dense(MAX_SEQUENCE * INPUT_DIM, activation="sigmoid"))
autoencoder.add(Reshape((MAX_SEQUENCE, INPUT_DIM)))


# TODO: learn how categorical_crossentropy and categorical_accuracy work
autoencoder.compile(optimizer='adam', loss=sequence_loss(), metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, sequence_accuracy])

checkpointer = ModelCheckpoint(filepath="models/sequence-model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# In the future, include batch size in steps_per_epoch
# QUESTION: what is the relation between batch size and the model? - I think it does not perform a
# gradient descent within a batch. Yes, seems like it. The gradient descent is done on vectors of shape
# (batch_size, INPUT_DIM)
# NOTE: for some reason the steps_per_epoch need to be 10 less than train_data//BATCH_SIZE. WHY?
history = autoencoder.fit_generator(train_data_generator.generate(), steps_per_epoch=(len(train_data)//BATCH_SIZE),
                        epochs=NUM_EPOCHS, validation_data=test_data_generator.generate(),
                        validation_steps=(len(test_data)//BATCH_SIZE), verbose=1,
                        callbacks=[checkpointer, tensorboard]).history
