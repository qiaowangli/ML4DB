

import numpy as np
from keras import models, layers, callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,SimpleRNN, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.losses import Loss
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from keras.optimizers import Adam
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge








class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if len(points) == 0:
        return None
    
    k = len(points[0])
    axis = depth % k
    
    points = sorted(points, key=lambda point: point[axis])
    median = len(points) // 2
    
    return KDNode(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def find_nearest_neighbor(root, target):
    best_distance = float('inf')
    best_point = None
    
    def search(node, target, depth):
        nonlocal best_distance, best_point
        
        if node is None:
            return
        
        axis = depth % len(target)
        current_point = node.point
        
        if euclidean_distance(target, current_point) < best_distance:
            best_distance = euclidean_distance(target, current_point)
            best_point = current_point
        
        if target[axis] < current_point[axis]:
            search(node.left, target, depth + 1)
        else:
            search(node.right, target, depth + 1)
        
        if abs(target[axis] - current_point[axis]) < best_distance:
            if target[axis] < current_point[axis]:
                search(node.right, target, depth + 1)
            else:
                search(node.left, target, depth + 1)
    
    search(root, target, 0)
    
    return best_point










class SmoothL1Loss(Loss):
    def __init__(self, delta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        smooth_loss = tf.where(abs_diff < self.delta, 0.5 * tf.square(abs_diff), abs_diff - 0.5 * self.delta)
        return tf.reduce_mean(smooth_loss)


class RelativeErrorLoss(tf.keras.losses.Loss):
    def __init__(self, tau = 0.5):
        super().__init__()
        self.tau = tau

    def call(self, y_true, y_pred):
        relative_error = tf.abs(((y_true - y_pred) + 0.00001) / (y_true + 0.00001))
        
        
        residual = y_true - y_pred
        loss = tf.maximum((self.tau - 1) * 10*(residual), self.tau * 10*(residual))

        total = loss * relative_error

        return tf.reduce_mean(total)



class WeightedMAELoss(tf.keras.losses.Loss):
    def __init__(self, sequence_length=50, weight_increase_rate=1.0):
        super().__init__()
        self.sequence_length = sequence_length
        self.weight_increase_rate = weight_increase_rate

    def call(self, y_true, y_pred):
        weights = tf.linspace(1.0, self.weight_increase_rate * self.sequence_length, self.sequence_length)
        
        error = tf.abs(y_true - y_pred)

        weighted_error = error * tf.expand_dims(weights, 0)  

        weighted_mae = tf.reduce_sum(weighted_error, axis=-1) / tf.reduce_sum(weights)
        
        return tf.reduce_mean(weighted_mae)

    




class QuantileRegressionLoss(tf.keras.losses.Loss):
    # the default is MAE / 2

    def __init__(self, tau = 0.5):
        super(QuantileRegressionLoss, self).__init__()
        self.tau = tau

    def call(self, y_true, y_pred):
        residual = y_true - y_pred
        loss = tf.maximum((self.tau - 1) * 10*(residual), self.tau * 10*(residual))
        return tf.reduce_mean(loss)




########## BASELINE MODEL.  ####################

def linear_regression(train_2d_data):
    X = []
    y = []
    for i in range(len(train_2d_data)-1):
        for j in range(len(train_2d_data[i])):
            X.append(train_2d_data[i])
            y.append(train_2d_data[i+1][j])

    model = LinearRegression()
    model.fit(X, y)
    return model

def kernel_regression(train_2d_data):
    X = []
    y = []
    for i in range(len(train_2d_data)-1):
        for j in range(len(train_2d_data[i])):
            X.append(train_2d_data[i])
            y.append(train_2d_data[i+1][j])

    model = KernelRidge()
    model.fit(X, y)
    return model







########## RNN MODEL.  ####################

def rnn_regression(feature_sequences, label_sequence):
    feature_sequences=np.array(feature_sequences, dtype=object) 
    label_sequence=np.array(label_sequence, dtype=object)

    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)


    rnn_cla_model = Sequential()
    rnn_cla_model.add(LSTM(120, activation="relu", input_shape=[len(x_train[0]), len(x_train[0][0])], return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(LSTM(100, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))

    # rnn_cla_model.add(LSTM(80, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # rnn_cla_model.add(Dropout(0.2))
    
    # rnn_cla_model.add(LSTM(60, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # rnn_cla_model.add(Dropout(0.2))

    # rnn_cla_model.add(LSTM(40, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # rnn_cla_model.add(Dropout(0.2))

    # rnn_cla_model.add(LSTM(20, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(LSTM(10, activation="relu", return_sequences=False, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # rnn_cla_model.add(Dropout(0.2))

    # rnn_cla_model.add(Dense(len(label_sequence[0]), activation='relu'))
    rnn_cla_model.add(Dense(len(label_sequence[0]), activation='linear'))

    # rnn_cla_model.compile(loss=WeightedMAELoss(weight_increase_rate=1.05), optimizer=Adam(), metrics=['mae'])
    rnn_cla_model.compile(loss='mae', optimizer=Adam(), metrics=['mae'])


    x_train=np.array(x_train, dtype=object)
    y_train=np.array(y_train, dtype=object)
    x_test=np.array(x_test, dtype=object)
    y_test=np.array(y_test, dtype=object) 
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test= x_test.astype(np.float32)
    y_test= y_test.astype(np.float32)

    EP = 10
    
    MLDB_model = rnn_cla_model.fit(x_train, y_train, validation_split=0.2,epochs=EP)

    loss_data = {
        'epoch': np.arange(1, int(EP+1)),
        'training_loss': MLDB_model.history['loss'],
        'validation_loss': MLDB_model.history['val_loss'],
    }

    df = pd.DataFrame(loss_data)
    df.to_csv('loss_data.csv', index=False)

    eva_res = rnn_cla_model.predict(x_test)

    with open('predictions.csv', 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)


      header = ['Sample'] + [f'Prediction_{i}' for i in range(1, eva_res.shape[1]+1)] + [f'Actual_{i}' for i in range(1, y_test.shape[1]+1)]
      csv_writer.writerow(header)

      for i, (prediction, actual_result) in enumerate(zip(eva_res, y_test)):
          row = [f'Test Sample {i+1}'] + prediction.tolist() + actual_result.tolist()
          csv_writer.writerow(row)


    return rnn_cla_model.evaluate(x_test, y_test)



def find_nearest_list_index(centerList, target_list):
   
    # KD TREE
    root = build_kdtree(centerList)
    nearest_neighbor = find_nearest_neighbor(root, target_list)
    return np.where(centerList == nearest_neighbor)[0][0]



