from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid
import keras.metrics as metrics
from keras.models import Model, load_model
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from autoencoder_helpers import card1_pred, card2_pred, card_pred

FEATURES = ['hole_1', 'hole_2', 'com_1', 'com_2', 'com_3', 'com_4', 'com_5']

def get_train_test_data():
    data = pd.read_csv('data.csv', skipinitialspace=True, skiprows=1, names=FEATURES)

    # One hot encode the card data
    data['hole_1'] = data['hole_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['hole_2'] = data['hole_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_1'] = data['com_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_2'] = data['com_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_3'] = data['com_3'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_4'] = data['com_4'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_5'] = data['com_5'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data = pd.get_dummies(data, prefix=['hole_1', 'hole_2', 'com_1', 'com_2', 'com_3', 'com_4', 'com_5'])

    # NOTE: deviating from reference implementation by not setting RADOM_SEED
    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))

input_dim = train_data.shape[1]
# 40-20 features give 100% accuracy at epoch 11
# 20-10 features give 100% accuracy at epoch 32
encoding_dim1 = 90 # 65 less than the number of features we have
encoding_dim2 = 30

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim1, activation='linear', activity_regularizer=regularizers.l1(10e-5))(input_layer)
# encoder = LeakyReLU(alpha=0.01)(encoder)
encoder = Dense(encoding_dim2, activation='linear')(encoder)
# encoder = LeakyReLU(alpha=0.01)(encoder)

decoder = Dense(encoding_dim1, activation='linear')(encoder)
# decoder = LeakyReLU(alpha=0.01)(decoder)
decoder = Dense(input_dim, activation='relu')(decoder)
# decoder = LeakyReLU(alpha=0.01)(decoder)
# decoder = sigmoid()(decoder)

# NOTE: using 'outputs' instead of 'output' in ref implementation because of updated API
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 32

# sgd = optimizers.SGD(lr=0.05, momentum=0.1, decay=0.0, nesterov=False)
adadelta = optimizers.Adadelta()

autoencoder.compile(optimizer=adadelta,
                    loss='binary_crossentropy',
                    metrics=[metrics.binary_accuracy, card_pred])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(train_data, train_data,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_data, test_data),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
