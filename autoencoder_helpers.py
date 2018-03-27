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

    card2_true = y_true[0:, 52:104]
    card2_pred = y_pred[0:, 52:104]

    card3_true = y_true[0:, 104:156]
    card3_pred = y_pred[0:, 104:156]

    card4_true = y_true[0:, 156:208]
    card4_pred = y_pred[0:, 156:208]

    card5_true = y_true[0:, 208:260]
    card5_pred = y_pred[0:, 208:260]

    card6_true = y_true[0:, 260:312]
    card6_pred = y_pred[0:, 260:312]

    card7_true = y_true[0:, 312:364]
    card7_pred = y_pred[0:, 312:364]

    card1 = K.equal(K.argmax(card1_true), K.argmax(card1_pred))
    card2 = K.equal(K.argmax(card2_true), K.argmax(card2_pred))
    card3 = K.equal(K.argmax(card3_true), K.argmax(card3_pred))
    card4 = K.equal(K.argmax(card4_true), K.argmax(card4_pred))
    card5 = K.equal(K.argmax(card5_true), K.argmax(card5_pred))
    card6 = K.equal(K.argmax(card6_true), K.argmax(card6_pred))
    card7 = K.equal(K.argmax(card7_true), K.argmax(card7_pred))

    return K.mean(K.concatenate([card1, card2, card3, card4, card5, card6, card7]), axis=-1)
