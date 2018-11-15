import pandas as pd
import numpy as np
import ast
import sys
import keras.metrics as metrics
from keras import regularizers
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, GRU, Reshape, Dense, Dropout, RepeatVector, Lambda, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
# from confusion_matrix_callback import ConfusionMatrix

NUM_EPOCHS = 100
BATCH_SIZE = 1
FEATURES = ['sequence']

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

def get_train_test_data():
    data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1, names=FEATURES).as_matrix()
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    return train_data, test_data

class KerasBatchGenerator(object):

    def __init__(self, data, batch_size):
        self.data = data
        # self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.currentIdx = 0

    # NOTE: not used because we decided not to use any padding
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

    # First, just return batch size 1
    def generate(self):
        while True:
            # print(self.data[self.currentIdx : self.currentIdx+self.batch_size][0])
            if self.currentIdx >= len(self.data):
                self.currentIdx = 0

            sequence = ast.literal_eval(self.data[self.currentIdx][0])
            sequences = [sequence]
            sequenceArr = np.array([sequence for sequence in sequences])
            self.currentIdx += 1

            yield sequenceArr, sequenceArr

def sequence_accuracy(y_true, y_pred):
    """
    A categorical_accuracy function that checks if the entire sequence is accurate
    instead of each action
    """
    return metrics.categorical_accuracy(y_true, y_pred)

def repeat_vector(args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

def train_model():
    input_dim = 5
    encoding_dim1 = 100
    encoding_dim2 = 30
    # encoding_dim2 = 3

    # # Model creation
    # autoencoder = Sequential()
    # # Encoder
    # autoencoder.add(LSTM(50, input_shape=(None, input_dim), return_sequences=False))
    # autoencoder.add(Dense(encoding_dim, activation="sigmoid"))
    # # In this repeat, we should get the time steps somehow
    # autoencoder.add(Lambda(repeat_vector, arguments={'layer_to_repeat': K.cast(autoencoder.layers[0].input_shape, K.floatx())}))
    # # Decoder
    # autoencoder.add(LSTM(input_dim, return_sequences=True))

    input = Input(shape=(None, input_dim))
    encoded = GRU(encoding_dim2, activation='sigmoid', return_sequences=False)(input)
    # encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
    # encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
    encoded = Dense(encoding_dim2, activation='sigmoid')(encoded)
    # decoded = Lambda(repeat_vector, output_shape=(None, encoding_dim1)) ([encoded, input])
    # decoded = LSTM(encoding_dim2, activation='sigmoid', return_sequences=False)(decoded)
    decoded = Lambda(repeat_vector, output_shape=(None, encoding_dim2)) ([encoded, input])
    decoded = GRU(input_dim, activation='sigmoid', return_sequences=True)(decoded)

    autoencoder = Model(input, decoded)

    # TODO: learn how categorical_crossentropy works
    autoencoder.compile(optimizer='adam', loss='categorical_hinge', metrics=[metrics.binary_accuracy, metrics.categorical_accuracy])

    return autoencoder

if len(sys.argv) < 3:
    autoencoder = train_model()
else:
    filename = sys.argv[2]
    autoencoder = load_model(filename)

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
train_data_generator = KerasBatchGenerator(train_data, BATCH_SIZE)
test_data_generator = KerasBatchGenerator(test_data, BATCH_SIZE)

checkpointer = ModelCheckpoint(filepath="models/sequence-model-{epoch:02d}.h5",
                               verbose=0,
                               save_best_only=True)
confMatrixData = test_data[:1000]
# confusionMatrix = ConfusionMatrix(confMatrixData)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# In the future, include batch size in steps_per_epoch
# QUESTION: what effect does batch size really have when backprop is not done between batches?
# NOTE: for some reason the steps_per_epoch need to be 10 less than train_data//BATCH_SIZE. WHY?
autoencoder.summary()
history = autoencoder.fit_generator(train_data_generator.generate(), steps_per_epoch=(len(train_data)//BATCH_SIZE) - 10,
                        epochs=NUM_EPOCHS, validation_data=test_data_generator.generate(),
                        validation_steps=(len(test_data)//BATCH_SIZE) - 10, verbose=1,
                        callbacks=[checkpointer, tensorboard]).history
