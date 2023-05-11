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

import random
import math

""" GLOBAL VARIABLE """


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
    dataSet['discrete_duration'] = pd.cut(dataSet['duration'], bins=[-2,0,100,200,300,400,500,600,700,800,900,dataSet['duration'].max()], labels=["C-1","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"])
    dataSet['template'] = dataSet['discrete_duration'].astype(str) +"<CHECKMARK>"+ dataSet['statement'].astype(str)


    """ Experimental configuration """
    # initial_template = dataSet['template'][1:700].unique()
    # print(len(initial_template))
    # seond_template = dataSet['template'][1:900].unique()
    final_template = dataSet['template'].unique()
    # print(final_template)


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
            while(sum(sequence_storage[query_group_index]) < random.randint(20, 50)):
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



def nn_setup(sequence_list, time_step):

    feature_sequences=[]
    label_sequence=[]
    index_point=0
    while index_point+time_step <= len(sequence_list)-1:
        feature_sequences.append(sequence_list[index_point:index_point+time_step])
        label_sequence.append(sequence_list[index_point+time_step])
        index_point+=1

    # print(len(feature_sequences))
    return feature_sequences, label_sequence


def generate_training_data_initial(sequence_list, initial_value):
    
    currentCutOff = initial_value
    # print(len(sequence_list))
    for currentListIndex in range(len(sequence_list)):
        if(sum(sequence_list[currentListIndex]) != sum(sequence_list[currentListIndex][:currentCutOff])):
            # print('dasdsa' +str(currentListIndex))
            return currentListIndex



def generate_training_data(sequence_list, initial_value, cutoffPrecentage, center):

    # Initial setup and center_construction #
    startingIndex = generate_training_data_initial(sequence_list,initial_value)
    trainningData, NN_Input_Center = center_construction(sequence_list[:startingIndex], initial_value, center)
    
    currentcutoffIndex = initial_value
    nextcutoffIndex = -1
    abnormalCounter = 0
    mainCounter = startingIndex
    # print(startingIndex)
    for currentListIndex in range(startingIndex+1, len(sequence_list)):
        if(sum(sequence_list[currentListIndex]) > sum(sequence_list[currentListIndex][:currentcutoffIndex])):
            abnormalCounter += 1
            nextcutoffIndex = max(nextcutoffIndex, [index for index, item in enumerate(sequence_list[currentListIndex]) if item != 0][-1]) + 1

        # print(len(center_matching(sequence_list[currentListIndex][:currentcutoffIndex], NN_Input_Center)))
        trainningData.append(center_matching(sequence_list[currentListIndex][:currentcutoffIndex], NN_Input_Center))
        mainCounter += 1

        if (abnormalCounter/mainCounter) >= cutoffPrecentage:
            # reconsturct 
            trainningData, NN_Input_Center = center_construction(sequence_list[:currentListIndex], nextcutoffIndex, center)
            # reset cutoff values
            currentcutoffIndex = nextcutoffIndex
            nextcutoffIndex = -1
            abnormalCounter = 0
        # print(abnormalCounter)
    # print(currentcutoffIndex)

    return trainningData

    
            


def center_construction(trainData, cutoffValue, center): 

    print(cutoffValue)
    truncated_list = [sublist[:cutoffValue] for sublist in trainData]
    trainData = truncated_list

    KmeanBuilder = kmean_cluster(trainData, k= center)
    
    kmean_model = KmeanBuilder.kmean()
    res = KmeanBuilder.classification(kmean_model)
    resC = KmeanBuilder.clusterCenter(kmean_model)  

    Input_Center=resC

    pcaCenter = PCA(center)
    NN_Input_Center=pcaCenter.fit_transform(Input_Center)

    # tsne = TSNE(n_components=80)
    # X_tsne = tsne.fit_transform(Input_Center)
    
    for currentIndex in range(len(trainData)):
        trainData[currentIndex] = NN_Input_Center[res[currentIndex]]
    
    return trainData, NN_Input_Center



def center_matching(target_list, centerList):
    closest_distance = math.inf
    closest_list = None
    for l in centerList:
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(l, target_list)]))
        if distance < closest_distance:
            closest_distance = distance
            closest_list = l
    return closest_list
