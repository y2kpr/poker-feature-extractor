from keras.callbacks import Callback
from sequence_autoencoder_verify import KerasBatchGenerator, custom_predict_generator

class ConfusionMatrix(Callback):
    def init(data):
        super(ModelCheckpoint, self).__init__()
        self.data = data

    def on_epoch_end(self, epoch, logs):
        dataGenerator = KerasBatchGenerator(data, BATCH_SIZE)
        cleanData = dataGenerator.get_data()

        predictions = custom_predict_generator(self.model, dataGenerator.generate(), steps=len(self.data))
        actionCM = action_confusion_matrix(cleanData, predictions)
        print(actionCM)
