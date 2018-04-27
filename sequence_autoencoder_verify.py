import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras.models import Sequential, load_model
from keras.layers import LSTM, Reshape
from keras.callbacks import ModelCheckpoint

NUM_EPOCHS = 10
MAX_SEQUENCE = 50
# FEATURES = ['bet', 'call', 'check', 'fold', 'raise']
FEATURES = ['sequence']

def get_train_test_data():
    data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1,
                names=FEATURES, nrows=10).as_matrix()

    return data

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
        paddedSequences = MAX_SEQUENCE - startIndex
        for i in range(0, paddedSequences):
            sequenceOutput = np.insert(sequenceOutput, startIndex + i, np.zeros(5), axis=1)

        # print('sequence is ' + str(sequence.shape))
        # print('sequence output is ' + str(sequenceOutput.shape))
        # print(sequence.shape)
        self.currentIdx += 1
        while True:
            yield sequence, sequenceOutput


data = get_train_test_data()
print('num of features is ' + str(data.shape))
data_generator = KerasBatchGenerator(data)
autoencoder = load_model('models/sequence-model-01.h5')
pd.set_option('display.max_rows', 1000)
print(data[8])

prediction = autoencoder.predict_generator(data_generator.generate(), steps=len(data))
print(prediction[8])
