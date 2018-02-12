import tensorflow as tf
import pandas as pd
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

FEATURES = ['community_1', 'community_2', 'community_3', 'community_4', 'community_5', 'hole_1', 'hole_2', 'round']

# def get_input_fn(data_set, num_epochs=None, shuffle=True):
#   return tf.estimator.inputs.pandas_input_fn(
#       x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
#       y = pd.Series(data_set[k].values for k in FEATURES),
#       num_epochs=num_epochs,
#       shuffle=shuffle)

def train_input_fn():
    # returns pandas DataFrame
    training_set = pd.read_csv('data.csv', skipinitialspace=True, skiprows=1, names=FEATURES)
    return pfe_input_fn(training_set)
    # print(training_set)

def pfe_input_fn(data_set):
    # For now, setting batch_size to the entire size of the data set
    batch_size = data_set.shape[0]
    # print("numpy arr is " + str(data_set.as_matrix()))
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set.as_matrix(), shape=[batch_size,8])
    return feature_cols, labels

# features: This is the first item returned from the input_fn passed to train,
# evaluate, and predict. This should be a single Tensor or dict of same.
# labels: This is the second item returned from the input_fn passed to train,
# evaluate, and predict. This should be a single Tensor or dict of same (for
# multi-head models). If mode is ModeKeys.PREDICT, labels=None will be passed.
# If the model_fn's signature does not accept mode, the model_fn must still be able to handle labels=None.
# mode: Optional. Specifies if this training, evaluation or prediction.
def pfe_model_fn(features, labels, mode):
    print('features are ' + str(features['round']))
    print('labels are ' + str(labels))

def main(unused_argv):
    # Create the Estimator
    feature_extractor = tf.estimator.Estimator(model_fn=pfe_model_fn)

    # Train the model
    feature_extractor.train(input_fn=train_input_fn, steps=100)

if __name__ == "__main__":
  tf.app.run()
