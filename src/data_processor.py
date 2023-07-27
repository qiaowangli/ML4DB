import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from cluster import kmean_cluster, Dbscan_cluster
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from collections import Counter


import random
import math

""" GLOBAL VARIABLE """

INITIAL_TEMPLATE_OBSERVATION = -1


def raw_data_processor(log_path,template_storage,sequence_storage,splitting_mode="query",predict_interval=6):
    """
    @input parameters:  log_path         -> log_file path
                        splitting_mode   -> splitting_mode decides the way to split the queries
                        predict_interval -> furture interval to predict
                        
    @output: template_storage -> A dictionary that contains all distint queries and their distint ID.
             sequence_storage -> A dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    """
    vaild_query_type=['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']

    dataSet = pd.read_csv(log_path)
    ########################################################################################################################
    dataSet['duration_checker'] = dataSet['message'].str.contains("duration")
    dataSet['duration'] = dataSet['message'].str.split('ms').str[0].str.split(" ").str[1]
    dataSet['duration'] = np.where(dataSet['duration_checker'] == True, dataSet['duration'], -1)
    dataSet['duration'] = dataSet['duration'].astype(float)
    dataSet['statement'] = dataSet['message'].str.split('ms').str[1].str.split("execute <unnamed>: ").str[1]
    # dataSet['discrete_duration'] = pd.cut(dataSet['duration'], bins=[-2,0,100,200,300,400,500,600,700,800,900,dataSet['duration'].max()], labels=["C-1","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"])
    dataSet['discrete_duration'] = pd.cut(dataSet['duration'], bins=[-2,0,500,900,dataSet['duration'].max()], labels=["C-1","C1","C2","C3"])
    dataSet['template'] = dataSet['discrete_duration'].astype(str) +"<CHECKMARK>"+ dataSet['statement'].astype(str)

    # dataSet['template'] = dataSet['statement'].astype(str)


    """ Experimental configuration """
    final_template = dataSet['template'].unique()
    print(len(final_template))
    # exit(-1)


    """ KEY: TEMPLATE , VALUE: INDEX OF THE TEMPLATE IN PREDICTION VECTOR"""
    TrackHash = {k: v for v, k in enumerate(final_template)}

    ########################################################################################################################
    query_group_index=0
    row_index_rawData = 0
    EOF_signal = False
    sequence_storage = []
    sequence_storage.append([0]*len(final_template))

    if splitting_mode == "query":
        while(row_index_rawData < len(dataSet)-1 or not EOF_signal):
            # print(row_index_rawData)
            density = random.randint(80, 100)
            while(sum(sequence_storage[query_group_index]) < density):
                sequence_storage[query_group_index][TrackHash[dataSet.iloc[row_index_rawData]['template']]] += 1
                if row_index_rawData == len(dataSet)-1:
                    EOF_signal = True
                    break
                else:
                    row_index_rawData+=1
            if not EOF_signal:  
                sequence_storage.append([0]*len(final_template))
                query_group_index+=1
  
    # print(len(sequence_storage))
    return TrackHash,sequence_storage



def nn_setup(sequence_list, time_step, NN_Input_Center, labels):

    feature_sequences=[]
    label_sequence=[]
    index_point=0
    while index_point+time_step <= len(sequence_list)-1:
        feature_sequences.append(sequence_list[index_point:index_point+time_step])
        label_sequence.append(NN_Input_Center[labels[index_point+time_step]])
        index_point+=1

    # print(len(feature_sequences))
    return feature_sequences, label_sequence


def generate_training_data_initial(sequence_list, initial_value):
       
    global INITIAL_TEMPLATE_OBSERVATION
    INITIAL_TEMPLATE_OBSERVATION = int(initial_value / 2)
    currentCutOff = initial_value
    for currentListIndex in range(len(sequence_list)):
        if(sum(sequence_list[currentListIndex]) != sum(sequence_list[currentListIndex][:currentCutOff])):
            print('starting at' + str(currentListIndex))
            return currentListIndex
    return len(sequence_list) - 1


def generate_training_data(sequence_list, initial_value, cutoffPrecentage, center):

    # Initial setup and center_construction #
    startingIndex = generate_training_data_initial(sequence_list,initial_value)
    trainningData, NN_Input_Center, labels = center_construction(sequence_list[:startingIndex], initial_value, center, startingIndex+1)
    
    currentcutoffIndex = initial_value
    nextcutoffIndex = -1
    abnormalCounter = 0
    mainCounter = startingIndex
    # print(startingIndex)
    for currentListIndex in range(startingIndex+1, len(sequence_list)):
        print('dsdsadsadasdsadsadasdasdasdasdasds')
        if(sum(sequence_list[currentListIndex]) > sum(sequence_list[currentListIndex][:currentcutoffIndex])):
            abnormalCounter += 1
            nextcutoffIndex = max(nextcutoffIndex, [index for index, item in enumerate(sequence_list[currentListIndex]) if item != 0][-1]) + 1

        trainningData.append(center_matching(sequence_list[currentListIndex][:currentcutoffIndex], NN_Input_Center))
        mainCounter += 1

        if (abnormalCounter/mainCounter) >= cutoffPrecentage:
            # reconsturct 
            trainningData, NN_Input_Center, labels = center_construction(sequence_list[:currentListIndex], nextcutoffIndex, center, currentListIndex)
            # reset cutoff values
            currentcutoffIndex = nextcutoffIndex
            nextcutoffIndex = -1
            abnormalCounter = 0


    return trainningData, NN_Input_Center, labels

    
            


def center_construction(trainData, cutoffValue, NumOfCenter, currentListIndex):

    truncated_list = [sublist[:cutoffValue] for sublist in trainData]
    trainData = truncated_list

    global INITIAL_TEMPLATE_OBSERVATION
    Pcaer = PCA(10)
    Pcaer=Pcaer.fit_transform(trainData)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(Pcaer)
    # Get the labels assigned by K-means to each data point
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    outList = []

    # centersNor = normalize_with_global_max(centers)

    centersNor = centers

    for currentIndex in range(len(labels)):
        if(labels[currentIndex] == -1):
            outList.append([0]*INITIAL_TEMPLATE_OBSERVATION)
        else:
            # outList.append(centersNor[labels[currentIndex]])
            outList.append(Pcaer[currentIndex])
    
    return outList, centersNor, labels


def center_matching(target_list, centerList):
    closest_distance = math.inf
    closest_list = None
    for l in centerList:
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(l, target_list)]))
        if distance < closest_distance:
            closest_distance = distance
            closest_list = l
    return closest_list


def normalize_with_global_max(centers):
    max_value = centers[0][0]
    min_value = 1000

    # Find the maximum value
    for row in centers:
        for value in row:
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value

    # Normalize each element
    for i in range(len(centers)):
        for j in range(len(centers[i])):
            centers[i][j] = ((centers[i][j] - min_value)/(max_value - min_value))

    return centers
