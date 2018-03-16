import keras.backend as K

# Shape of y_true is (batch_size, number of features)
def card1_pred(y_true, y_pred):
    card1_true = y_true[0:, 0:52]
    card1_pred = y_pred[0:, 0:52]
    return K.mean(K.equal(K.argmax(card1_true), K.argmax(card1_pred)), axis=-1)

def card2_pred(y_true, y_pred):
    card2_true = y_true[0:, 52:105]
    card2_pred = y_pred[0:, 52:105]
    return K.mean(K.equal(K.argmax(card2_true), K.argmax(card2_pred)), axis=-1)

def card_pred(y_true, y_pred):
    card1_true = y_true[0:, 0:52]
    card1_pred = y_pred[0:, 0:52]
    card2_true = y_true[0:, 52:105]
    card2_pred = y_pred[0:, 52:105]

    return K.mean(K.equal(K.argmax(card1_true), K.argmax(card1_pred)), axis=-1)
