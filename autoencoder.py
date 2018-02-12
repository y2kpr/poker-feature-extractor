from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import pandas as pd

FEATURES = ['community_1', 'community_2', 'community_3', 'community_4', 'community_5', 'hole_1', 'hole_2', 'round']

data = pd.read_csv('data.csv', skipinitialspace=True, skiprows=1, names=FEATURES)
# NOTE: deviating from reference implementation by not setting RADOM_SEED
train_data, test_data = train_test_split(data, test_size=0.2)

print('num of features is ' + str(train_data.shape[1]))

input_dim = train_data.shape[1]
encoding_dim1 = 7 # 1 less than the number of features we have
encoding_dim2 = 4

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim1, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(encoding_dim2, activation='relu')(encoder)

decoder = Dense(encoding_dim2, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

# NOTE: using 'outputs' instead of 'output' in ref implementation because of updated API
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 32

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

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
