#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3

import numpy as np
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def lstm_predictor(feature_sequences, label_sequence,num_folds=-1, baseline_test=False):
    """
    This function provides MANY-TO-ONE RNN prediction, if "num_folds" is -1, the function would not use cross-validation, if "num_folds" != -1, the function would use "num_folds" folders cross-validation
    """
    # convert to numpy array
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)
    if not baseline_test: 
        if num_folds != -1:
            folder_list=[]
            kfold = KFold(n_splits=num_folds, shuffle=True)

            for train, test in kfold.split(feature_sequences, label_sequence):
                lstm_model = Sequential()
                lstm_model.add(LSTM(100, activation="relu", input_shape=feature_sequences[train].shape[1:], return_sequences=True ))
                lstm_model.add(Dropout(0.2))

                lstm_model.add(LSTM(50, activation="relu", return_sequences=True ))
                lstm_model.add(Dropout(0.2))

                lstm_model.add(LSTM(30, activation="relu", return_sequences=False ))
                lstm_model.add(Dropout(0.2))

                lstm_model.add(Dense(len(label_sequence[0]), activation="relu", input_shape=feature_sequences[train].shape[1:]))

                lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

                # now lets train our model
                lstm_model.fit(feature_sequences[train], label_sequence[train], epochs=10)

                # now lets validate our model
                folder_list.append(lstm_model.evaluate(feature_sequences[test], label_sequence[test])[1])

            return sum(folder_list)/len(folder_list)
        else:
            x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)
            lstm_model = Sequential()
            lstm_model.add(SimpleRNN(100, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
            lstm_model.add(Dropout(0.2))

            lstm_model.add(SimpleRNN(60, activation="relu", return_sequences=True ))
            lstm_model.add(Dropout(0.2))

            lstm_model.add(SimpleRNN(30, activation="relu", input_shape=x_train.shape[1:], return_sequences=False ))
            lstm_model.add(Dropout(0.2))

            lstm_model.add(Dense(len(label_sequence[0]), activation="relu"))

            lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            # now lets train our model
            lstm_model.fit(x_train, y_train, epochs=10)

            # now lets validate our model
            return lstm_model.evaluate(x_test, y_test)[1]
    else:
        """
        baseline test(FNN)
        """
        x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)

        fnn_model = Sequential()

        fnn_model.add(Dense(100, activation="relu", input_shape=x_train.shape[1:]))
        fnn_model.add(Dropout(0.2))
        fnn_model.add(Dense(50, activation="relu"))
        fnn_model.add(Dropout(0.2))
        fnn_model.add(Dense(30, activation="relu"))
        fnn_model.add(Dropout(0.2))
        fnn_model.add(Dense(len(label_sequence[0]), activation="relu"))
        fnn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        # now lets train our model
        fnn_model.fit(x_train, y_train, epochs=10)
        # return the prediction
        return fnn_model.evaluate(x_test, y_test)[1]


  

def manual_test(feature_sequences, label_sequence):
    # convert to numpy array
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)
    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)

    lstm_model = Sequential()

    lstm_model.add(LSTM(30, activation="relu", input_shape=x_train.shape[1:], return_sequences=False ))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(len(label_sequence[0]), activation="relu", input_shape=x_train.shape[1:]))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # now lets train our model
    lstm_model.fit(x_train, y_train, epochs=10)
    # return the prediction
    return lstm_model.predict(x_test)[0],lstm_model.predict(x_test)[0]

def rnn_classification(feature_sequences, label_sequence,center_list=None,baseline_test=False):
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)
    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)
    rnn_cla_model = Sequential()
    rnn_cla_model.add(LSTM(90, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    rnn_cla_model.add(Dropout(0.2))
    rnn_cla_model.add(LSTM(60, activation="relu", return_sequences=True ))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(LSTM(30, activation="relu", return_sequences=False ))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(Dense(len(label_sequence[0]), activation="softmax"))

    rnn_cla_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # now lets train our model
    rnn_cla_model.fit(x_train, y_train, epochs=10)
    return rnn_cla_model.evaluate(x_test, y_test)[1]
