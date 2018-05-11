import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras.models import Sequential, load_model
from keras.layers import LSTM, Reshape
from keras.callbacks import ModelCheckpoint

NUM_EPOCHS = 10
MAX_SEQUENCE = 50
BATCH_SIZE = 1
# FEATURES = ['bet', 'call', 'check', 'fold', 'raise']
FEATURES = ['sequence']

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
            self.currentIdx += self.batch_size
            print(sequencesArr.shape)
            print(maxLength)
            yield sequencesArr, sequencesArr


data = pd.read_csv('sequence_data.csv', skipinitialspace=True, skiprows=1,
            names=FEATURES, nrows=10).as_matrix()
print('num of features is ' + str(data.shape))
data_generator = KerasBatchGenerator(data, BATCH_SIZE)
autoencoder = load_model('models/sequence-model.h5')
pd.set_option('display.max_rows', 1000)
print(data[8])

prediction = autoencoder.predict_generator(data_generator.generate(), steps=len(data)//BATCH_SIZE)
print(prediction[8])
