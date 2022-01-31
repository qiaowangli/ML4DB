#!/usr/bin/python3

import numpy as np
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split


def lstm_predictor(feature_sequences, label_sequence):
    # convert to numpy array
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)

    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.3, random_state=0)

    lstm_model = Sequential()
    lstm_model.add(LSTM(500, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    lstm_model.add(Dropout(0.2))

    lstm_model.add(LSTM(250, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    lstm_model.add(Dropout(0.2))

    lstm_model.add(LSTM(100, activation="relu", input_shape=x_train.shape[1:], return_sequences=False ))
    lstm_model.add(Dropout(0.2))

    lstm_model.add(Dense(len(label_sequence[0]), activation="relu", input_shape=x_train.shape[1:]))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # now lets train our model
    lstm_model.fit(x_train, y_train, epochs=10)

    # now lets validate our model
    return lstm_model.evaluate(x_test,y_test)


