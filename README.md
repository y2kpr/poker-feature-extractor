# Description

Neural network to learn useful features from raw poker game state.

# How to run

Make sure you have tensorflow, keras, numpy installed (you may be missing something else too).

### To train the card model:

1. `cd card_nn`
2. Generate the data by running: `python card_data_generator.py <sample size>`
3. Train the model by running: `python card_autoencoder.py`
4. Find the trained model by epoch in the `models/` directory.

### To train the action sequence model:

1. `cd action_sequence_nn/`
2. Generate the data by running: `python sequence_data_generator.py <sample size>`
3. Train the model by running: `python sequence_autoencoder.py`
4. Find the trained model by epoch in the `models/` directory.
