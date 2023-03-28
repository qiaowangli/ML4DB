

import numpy as np
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,SimpleRNN
from sklearn.model_selection import train_test_split



def rnn_regression(feature_sequences, label_sequence,centerList):
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)

    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)

    print(x_train.shape[1:])
    print(y_train[0])
    rnn_cla_model = Sequential()
    rnn_cla_model.add(LSTM(90, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(LSTM(60, activation="relu", return_sequences=False ))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(Dense(len(label_sequence[0], activation='linear')))

    rnn_cla_model.compile(loss='mean_squared_error', optimizer='adam')
    # # now lets train our model
    rnn_cla_model.fit(x_train, y_train, epochs=10)

    yPredict= rnn_cla_model.predict(x_test)

    correctMatching = 0
    ########## backward matching evaluation ################
    for index in y_test:
        if centerList.index(y_test[index]) == find_nearest_list_index(centerList,yPredict[index]):
            correctMatching+=1
    
    print(correctMatching/len(y_test))

    

    return rnn_cla_model.evaluate(x_test, y_test)[1]


def euclidean_distance(a, b):

    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))

def find_nearest_list_index(centerList, target_list):

    nearest_index = 0
    nearest_distance = euclidean_distance(centerList[0], target_list)

    for i in range(1, len(centerList)):
        distance = euclidean_distance(centerList[i], target_list)
        if distance < nearest_distance:
            nearest_index = i
            nearest_distance = distance

    return nearest_index
