import pandas as pd
import numpy as np
import ast
import keras.metrics as metrics
from keras.models import load_model, Model
from keras.layers import LSTM, Reshape
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from card_nn.card_autoencoder_helpers import card_pred, card1_pred, card2_pred

FEATURES = ['action_sequences', 'card_sequences']

sequence_autoencoder = load_model('action_sequence_nn/models/97%_GRU_30_features.h5')
card_autoencoder = load_model('card_nn/models/100%_no_zeroes.h5', custom_objects={'card_pred': card_pred, 'card1_pred': card1_pred,
    'card2_pred': card2_pred})

sequence_encoder = Model(input=sequence_autoencoder.input,
    output=sequence_autoencoder.get_layer(name='dense_1').output)
card_encoder = Model(input=card_autoencoder.input,
    output=card_autoencoder.get_layer(name='dense_3').output)

game_data = pd.read_csv('game_data.csv', skipinitialspace=True, skiprows=1,
            names=FEATURES, nrows=1).as_matrix()

# just get one game's action sequence data
sequence_data = np.array([ast.literal_eval(game_data[0][0])])
# print(sequence_data)
cards_data = ast.literal_eval(game_data[0][1])
# print(cards_data)

running_sequence = None
full_encodings = []
for pos, action in enumerate(sequence_data[0]):
    # TODO: figure out a better way to initilaze running_sequence
    # ENCODE SEQUENCE
    if pos == 0:
        running_sequence = np.reshape(action, (1,5))
    else:
        # print(action)
        running_sequence = np.concatenate((running_sequence, np.reshape(action, (1, 5))))
    # print(running_sequence)
    sequence_encoding = sequence_encoder.predict(np.reshape(running_sequence, (1,len(running_sequence),5)))
    print('Encodings at action ' + str(pos+1) + ':')
    # print(sequence_encoding)

    # ENCODE CARDS
    cards = cards_data[pos]
    # one hot encode each card
    CARD_SIZE = 52
    one_hot_cards = np.zeros((1,len(cards) * CARD_SIZE))
    # print(one_hot_cards.shape)
    for pos, card in enumerate(cards):
        # If card is not drawn yet, do not encode it (leave it as all zeros)
        if card != 0:
            one_hot_cards[0][pos*CARD_SIZE + card] = 1

    # print(one_hot_cards)
    # Pass through model to get encodings
    cards_encoding = card_encoder.predict(one_hot_cards)
    # print(cards_encoding)
    # full_encodings.append(np.append(sequence_encoding, cards_encoding))
    full_encodings.append(sequence_encoding)

print(full_encodings)
# perform dimensionality reduction
pca = PCA(n_components=2)
pca_encodings = pca.fit_transform(full_encodings[0])
print(pca_encodings)
