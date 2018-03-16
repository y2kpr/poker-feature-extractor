from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
from autoencoder_helpers import card1_pred
from autoencoder_helpers import card2_pred
# from autoencoder import get_train_test_data

FEATURES = ['hole_1', 'hole_2']

def get_train_test_data():
    data = pd.read_csv('predict.csv', skipinitialspace=True, skiprows=1, names=FEATURES, nrows=100)

    # One hot encode the card data
    data['hole_1'] = data['hole_1'].astype('category', categories=list(range(1,53)))
    data['hole_2'] = data['hole_2'].astype('category', categories=list(range(1,53)))
    data = pd.get_dummies(data, prefix=['hole_1', 'hole_2'])

    return data

autoencoder = load_model('model.h5', custom_objects={'card1_pred': card1_pred, 'card2_pred': card2_pred})
data = get_train_test_data()

pd.set_option('display.max_rows', 1000)
print(data.iloc[80])

prediction = autoencoder.predict(data)

print(prediction[80])
