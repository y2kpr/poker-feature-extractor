from keras.models import Model, load_model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from autoencoder_helpers import card_pred, card1_pred, card2_pred
# from autoencoder import get_train_test_data

FEATURES = ['hole_1', 'hole_2', 'com_1', 'com_2', 'com_3', 'com_4', 'com_5']

def get_train_test_data():
    data = pd.read_csv('predict.csv', skipinitialspace=True, skiprows=1, names=FEATURES, nrows=100)

    # One hot encode the card data
    data['hole_1'] = data['hole_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['hole_2'] = data['hole_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_1'] = data['com_1'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_2'] = data['com_2'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_3'] = data['com_3'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_4'] = data['com_4'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data['com_5'] = data['com_5'].astype(pd.api.types.CategoricalDtype(categories=list(range(1,53))))
    data = pd.get_dummies(data, prefix=['hole_1', 'hole_2', 'com_1', 'com_2', 'com_3', 'com_4', 'com_5'])

    return data

def verify():
    autoencoder = load_model('model.h5', custom_objects={'card_pred': card_pred, 'card1_pred': card1_pred,
        'card2_pred': card2_pred})
    data = get_train_test_data()
    pd.set_option('display.max_rows', 1000)
    print(data.iloc[80])

    prediction = autoencoder.predict(data)
    print(prediction[80])

def print_intermediate():
    autoencoder = load_model('model.h5', custom_objects={'card_pred': card_pred})
    data = get_train_test_data()

    input_dim = data.shape[1]
    # 40-20 features give 100% accuracy at epoch 11
    # 20-10 features give 100% accuracy at epoch 32
    encoding_dim1 = 90 # 65 less than the number of features we have
    encoding_dim2 = 30

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim1, weights=autoencoder.layers[1].get_weights())(input_layer)
    # encoder = LeakyReLU(alpha=0.01)(encoder)
    encoder = Dense(encoding_dim2, weights=autoencoder.layers[2].get_weights())(encoder)
    # encoder = LeakyReLU(alpha=0.01)(encoder)

    autoencoder2 = Model(inputs=input_layer, outputs=encoder)


    prediction = autoencoder2.predict(data)
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    print(prediction)

verify()
