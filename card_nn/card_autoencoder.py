from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid
import keras.metrics as metrics
import keras.backend as K
from keras.models import Model, load_model
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from card_autoencoder_helpers import card1_pred, card2_pred, card_pred

FEATURES = ['hole_1', 'hole_2', 'com_1', 'com_2', 'com_3', 'com_4', 'com_5']

# Activate TF_DBG (tensorflow debugger)
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

def get_train_test_data():
    data = pd.read_csv('card_data.csv', skipinitialspace=True, skiprows=1, names=FEATURES)

    # One hot encode the card data
    data['hole_1'] = data['hole_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['hole_2'] = data['hole_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_1'] = data['com_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_2'] = data['com_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_3'] = data['com_3'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_4'] = data['com_4'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_5'] = data['com_5'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    # data['round'] = data['round'].astype(pd.api.types.CategoricalDtype(categories=list(range(0,4))))

    data = pd.get_dummies(data, prefix=['com_1', 'com_2', 'com_3', 'com_4', 'com_5', 'hole_1', 'hole_2'])

    # NOTE: deviating from reference implementation by not setting RADOM_SEED
    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data

train_data, test_data = get_train_test_data()
print('num of features is ' + str(train_data.shape))
print(train_data)

input_dim = train_data.shape[1]
# 60-120 features give 100% accuracy
# 45-90 features give 99.2% accuracy at epoch 4
# (sigmoid) 25 features 100% accuracy at epoch 3
# (sigmoid) 20 features 100% accuract at epoch 89
encoding_dim1 = 100 # 65 less than the number of features we have
encoding_dim2 = 50
encoding_dim3 = 20

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim1, activation='linear')(input_layer)
# encoder = LeakyReLU(alpha=0.01)(encoder)
encoder = Dense(encoding_dim2, activation='linear')(encoder)
encoder = Dense(encoding_dim3, activation='linear')(encoder) # Apply sigmoid activation here too or else it is just a linear regression algo
# encoder = LeakyReLU(alpha=0.01)(encoder)

decoder = Dense(encoding_dim2, activation='linear')(encoder)
decoder = Dense(encoding_dim1, activation='linear')(decoder)
# decoder = LeakyReLU(alpha=0.01)(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
# decoder = LeakyReLU(alpha=0.01)(decoder)

# NOTE: using 'outputs' instead of 'output' in ref implementation because of updated API
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 32

# sgd = optimizers.SGD(lr=0.05, momentum=0.1, decay=0.0, nesterov=False)
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[metrics.binary_accuracy, card_pred])

checkpointer = ModelCheckpoint(filepath="models/card-model-{epoch:02d}.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
autoencoder.summary()
history = autoencoder.fit(train_data, train_data,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_data, test_data),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
