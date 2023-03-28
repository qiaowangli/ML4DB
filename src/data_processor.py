
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from cluster import kmean_cluster, Dbscan_cluster
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
    initial_template = dataSet['template'][1:700].unique()
    print(len(initial_template))
    seond_template = dataSet['template'][1:900].unique()
    final_template = dataSet['template'].unique()

    """ KEY: TEMPLATE , VALUE: INDEX OF THE TEMPLATE IN PREDICTION VECTOR"""
    TrackHash = {k: v for v, k in enumerate(final_template)}

    # print(dataSet.iloc[0])
    ########################################################################################################################
    query_group_index=0
    row_index_rawData = 0
    EOF_signal = False
    sequence_storage = []
    sequence_storage.append([0]*len(final_template))

    if splitting_mode == "query":
        while(row_index_rawData < len(dataSet) or EOF_signal):
            while(sum(sequence_storage[query_group_index]) < predict_interval):
                sequence_storage[query_group_index][TrackHash[dataSet.iloc[row_index_rawData]['template']]] += 1
                if row_index_rawData == len(dataSet):
                    EOF_signal = True
                    break
                else:
                    row_index_rawData+=1
            if not EOF_signal:  
                if row_index_rawData < 700:
                    sequence_storage.append([0]*len(final_template))
                elif row_index_rawData < 900:
                    sequence_storage.append([0]*len(final_template))
                else:
                    sequence_storage.append([0]*len(final_template))
                query_group_index+=1
  
    return TrackHash,sequence_storage



def nn_setup(sequence_list, time_step):
    """
    time_step indicates the step wise of RNN, if time_step is 1, the function generates the dataset for FNN.
    """
    feature_sequences=[]
    label_sequence=[]
    # total_sequence=[]
    index_point=0
    while index_point+time_step <= len(sequence_list)-1:
        feature_sequences.append(sequence_list[index_point:index_point+time_step])
        label_sequence.append(sequence_list[index_point+time_step])
        index_point+=1

    return feature_sequences, label_sequence,sequence_list

def generate_training_data(feature_sequences, label_sequences,sequence_list):
    
    # x_train, x_test, y_train, y_test = train_test_split(feature_sequences, label_sequences, test_size=0.3, random_state=0, shuffle=False)
    initial_value = 99
    trainData = sequence_list[:50]

    new_list = [row[:initial_value] for row in trainData]
    trainData = new_list
    KmeanBuilder = kmean_cluster(trainData[0:initial_value], k= 5, pcaValue=60)
    # KmeanBuilder.pca(sequence_list)
    kmean_model = KmeanBuilder.kmean()
    res = KmeanBuilder.classification(kmean_model)
    resC = KmeanBuilder.clusterCenter(kmean_model)  
    pca = PCA(0.98)
    NN_Input_Center=pca.fit_transform(resC)
    # now we get the centers where the length of all centers must be the same
    centerList = {}
    centerIndex =0
    for centerIndex in range(len(NN_Input_Center)):
        centerList[centerIndex] = NN_Input_Center[centerIndex]

    #now we get the hashtable to convert our prediction vector to the coressponding center

    for pIndex in range(len(feature_sequences)):
        for lIndex in range(len(feature_sequences[pIndex])):
            feature_sequences[pIndex][lIndex] = centerList[KmeanBuilder.clusterPredict(kmean_model,[feature_sequences[pIndex][lIndex][:initial_value]])[0]]
    
    for pIndex in range(len(label_sequences)):
        label_sequences[pIndex] = centerList[KmeanBuilder.clusterPredict(kmean_model,[label_sequences[pIndex][:initial_value]])[0]]
    
    return feature_sequences,label_sequences, centerList

    



