import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras.models import Sequential, load_model
from keras.layers import LSTM, Reshape
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 1
NUM_PREDICTIONS = 10000
FEATURES = ['sequence']

class KerasBatchGenerator(object):

    def __init__(self, data, batch_size):
        self.data = self.to_list(data)
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.currentIdx = 0

    def to_list(self, data):
        for i in range(len(data)):
            data[i] = [ast.literal_eval(data[i][0])]
        return data

    def get_data(self):
        return self.data

    # First, just return batch size 1
    def generate(self):
        while True:
            # print(self.data[self.currentIdx : self.currentIdx+self.batch_size][0])
            if self.currentIdx >= len(self.data):
                self.currentIdx = 0

            # sequence = ast.literal_eval(self.data[self.currentIdx][0])
            # sequences = [sequence]
            sequenceArr = np.array([sequence for sequence in self.data[self.currentIdx]])
            self.currentIdx += 1

            yield sequenceArr, sequenceArr

def custom_predict_generator(model, generator, steps):
    stepsDone = 0
    allOuts = []

    while stepsDone < steps:
        # assuming a tuple with (inputs, targets). Ingore the labels
        data, _ = next(generator)
        outs = model.predict_on_batch(data)
        stepsDone += 1
        allOuts.append(outs[0])
    # allOuts = np.array(allOuts)

    return allOuts

def action_confusion_matrix(trueSeqs, predSeqs):
    # flatten lists to lose sequence information because we only care about individual actions
    trueActions = [action for sequence in trueSeqs for action in sequence[0]]
    trueActions = [np.argmax(action) for action in trueActions]
    predActions = [action for sequence in predSeqs for action in sequence]
    predActions = [np.argmax(action) for action in predActions]

    actionCM = confusion_matrix(trueActions, predActions)

    # print some stats
    print("num of total actions: " + str(len(trueActions)))
    numErrors = 0
    for i in range(len(actionCM)):
        for j in range(len(actionCM)):
            if i == j:
                continue
            numErrors += actionCM[i][j]
    print("num of wrongly predicted action: " + str(numErrors))
    accuracy = 1-(numErrors/len(trueActions))
    print("accuracy: " + str(accuracy*100) + "%")

    return actionCM

data = pd.read_csv('sequence_data_verify.csv', skipinitialspace=True, skiprows=1,
            names=FEATURES, nrows=NUM_PREDICTIONS).as_matrix()
print('num of features is ' + str(data.shape))
dataGenerator = KerasBatchGenerator(data, BATCH_SIZE)
autoencoder = load_model('models/97.8%_GRU_50_features.h5')
pd.set_option('display.max_rows', 1000)
cleanData = dataGenerator.get_data()

predictions = custom_predict_generator(autoencoder, dataGenerator.generate(), steps=NUM_PREDICTIONS)
actionCM = action_confusion_matrix(cleanData, predictions)
print(actionCM)

# print(predictions[0])
