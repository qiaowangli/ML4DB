

import numpy as np
from keras import models, layers, callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.losses import Loss
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from keras.optimizers import Adam




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



class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class SmoothL1Loss(Loss):
    def __init__(self, delta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        smooth_loss = tf.where(abs_diff < self.delta, 0.5 * tf.square(abs_diff), abs_diff - 0.5 * self.delta)
        return tf.reduce_mean(smooth_loss)

class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        quadratic_loss = 0.5 * tf.square(error)
        linear_loss = self.delta * (tf.abs(error) - 0.5 * self.delta)
        huber_loss = tf.where(tf.abs(error) <= self.delta, quadratic_loss, linear_loss)
        return tf.reduce_mean(huber_loss)




def rnn_regression(feature_sequences, label_sequence,centerList):
    feature_sequences=np.array(feature_sequences) 
    label_sequence=np.array(label_sequence)

    x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequence, test_size=0.2, random_state=0)

    print(x_train.shape[1:])
    print(y_train[0])
    rnn_cla_model = Sequential()
    rnn_cla_model.add(SimpleRNN(120, activation="relu", input_shape=x_train.shape[1:], return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(SimpleRNN(100, activation="relu", input_shape=x_train.shape[1:], return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))

    rnn_cla_model.add(SimpleRNN(80, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))
    
    rnn_cla_model.add(SimpleRNN(60, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(SimpleRNN(40, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(SimpleRNN(20, activation="relu", return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(SimpleRNN(10, activation="relu", return_sequences=False, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    rnn_cla_model.add(Dropout(0.2))

    rnn_cla_model.add(Dense(len(label_sequence[0]), activation='linear'))

    rnn_cla_model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['mae'])



    rnn_cla_model.fit(x_train, y_train, epochs=100, batch_size = 16)

    yPredict= rnn_cla_model.predict(x_test)

    track1 = {}
    track2 = {}


    correctMatching = 0
    ########## backward matching evaluation ################
    for index in range(len(y_test)):
        act = np.where(centerList == y_test[index])[0][0]
        pre = find_nearest_list_index(centerList,yPredict[index])
        if act not in track1:
          track1[act] = 1
        else:
          track1[act] += 1

        if pre not in track2:
          track2[pre] = 1
        else:
          track2[pre] += 1

        if act == pre:
            correctMatching+=1
        
    print('the result acc is ' + str(correctMatching/len(y_test)))
    print(track1)
    print(track2)

    

    return rnn_cla_model.evaluate(x_test, y_test)



def find_nearest_list_index(centerList, target_list):
   
    # KD TREE
    root = build_kdtree(centerList)
    nearest_neighbor = find_nearest_neighbor(root, target_list)
    return np.where(centerList == nearest_neighbor)[0][0]



