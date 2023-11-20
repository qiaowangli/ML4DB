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
import matplotlib.pyplot as plt
import random
import math
from collections import Counter




def extract_stable_top_rank(inputList, observation_slot = 1000, TOP_RANK = 10):
    # print(count_element_frequency_in_2d_list(inputList[0]))
    # The return is a set, where the length of set is TOP_RANK

    result_dict = {element: 0 for sublist in inputList for subsubList in sublist for element in subsubList}

    observation_slot = observation_slot if observation_slot <= len(inputList) else len(inputList)
    for sublist in inputList[:observation_slot]:
        for subsubList in sublist:
            for element in subsubList:
                result_dict[element] += 1

    TOP_RANK = TOP_RANK if TOP_RANK <= len(result_dict) else len(result_dict)
    return sorted(result_dict, key=lambda x: int(result_dict[x]), reverse=True)[:TOP_RANK]
    




def split_list_into_chunks(A, k, stepK = True):
    if k <= 0 or not isinstance(k, int):
        raise ValueError("k must be a positive integer")

    if len(A) < k:
        raise ValueError("k must be less than or equal to the length of the input list")
    
    if stepK:
        A = A.tolist()  # Convert the Pandas Series to a Python list

    result = [A[i:i + k] for i in range(len(A) - k + 1)]

    if stepK:
        return result, set(A)
    else:
        return result



def count_element_frequency_in_2d_list(A):

    result_dict = {element: [] for sublist in A for element in sublist}

    keys_list = list(result_dict.keys())

    for lst in A:
        for element in keys_list:
                result_dict[element].append(lst.count(element))
    return result_dict






def analyze_top_keys_variation(data_list, TOP_RANK):

    TOP_RANK = TOP_RANK
    variations = []
    resList = []

    for data in data_list:
        input_dict = count_element_frequency_in_2d_list(data)
        top_5_keys = sorted(input_dict, key=lambda x: sum(input_dict[x]), reverse=True)[:TOP_RANK]
        
        if variations:
            common_keys = set(variations[-1]).intersection(top_5_keys)
            # print(common_keys)
            variation_count = TOP_RANK - len(common_keys)
        else:
            variation_count = 0

        variations.append(top_5_keys)
        resList.append(variation_count)

        # print("Top 5 Keys:", top_5_keys)
        print("Variation Count:", variation_count)
        print("---------------------------------")
    
    x_axis = range(1, len(data_list) + 1)
    plt.plot(x_axis, resList, marker='o')
    plt.xlabel('Data Group')
    plt.ylabel('Top  Keys Variation Count')
    plt.title('Top {} Keys Variation in {} Groups'.format(TOP_RANK, len(data_list)))
    plt.show()


def create_NN_input(data_list, TOP_RANK):

    TOP_RANK = TOP_RANK
    singleTreatment = []
    returnList = []

    for data in data_list:
        input_dict = count_element_frequency_in_2d_list(data)
        top_keys = sorted(input_dict, key=lambda x: sum(input_dict[x]), reverse=True)[:TOP_RANK]
        singleTreatment = []
        for singleVector in data:
            tmpList = []
            for top_key in top_keys:
                tmpList.append(singleVector.count(top_key))
            while len(tmpList) < TOP_RANK:
                tmpList.append(0)
            singleTreatment.append(tmpList)
        returnList.append(singleTreatment) 
    return returnList


def create_NN_input_with_constant_TOP_RANK(data_list, TOP_RANK):
    top_keys = extract_stable_top_rank(data_list, 10000, TOP_RANK)
    top_keys_set = set(top_keys)

    singleTreatment = []
    returnList = []

    for data in data_list:
        singleTreatment = []
        for singleVector in data:
            # tmpList = []
            vector_counter = Counter(singleVector) 
            tmpList = [vector_counter[key] for key in top_keys if key in top_keys_set]
            # for top_key in top_keys:
            #     tmpList.append(singleVector.count(top_key))
            singleTreatment.append(tmpList)
        returnList.append(singleTreatment) 
    return returnList



            
        

def raw_data_processor(log_path, K, G, TOP_RANK, isASmallTest):
    """
    @input parameters:  log_path         -> log_file path
                        splitting_mode   -> splitting_mode decides the way to split the queries
                        predict_interval -> furture interval to predict
                        
    @output: template_storage -> A dictionary that contains all distint queries and their distint ID.
             sequence_storage -> A dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    """
    vaild_query_type=['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']

    dataSet = pd.read_csv(log_path)
    print(len(dataSet))
    ########################################################################################################################

    # Step 1: Extract 'duration' and 'statement' columns
    extracted_data = dataSet['message'].str.extract(r'duration:\s*(\d+(\.\d+)?)\s+ms\s+execute(.*)', expand=True)
    dataSet['duration'] = extracted_data[0].fillna(0)
    dataSet['statement'] = extracted_data[2].fillna("MARKER")

    # Step 2: Convert 'duration' to float and replace NaN with -1
    dataSet['duration'] = dataSet['duration'].astype(float)

    # Step 3: Create 'discrete_duration' column using pd.cut()
    bins = [-2, 0, 500, 900, dataSet['duration'].max()]
    # print(dataSet['duration'].max())
    labels = ["C-1", "C1", "C2", "C3"]
    dataSet['discrete_duration'] = pd.cut(dataSet['duration'], bins=bins, labels=labels)

    # Step 4: Create 'template' column by combining 'discrete_duration' and 'statement'
    dataSet['template'] = dataSet['discrete_duration'].astype(str) + "<CHECKMARK>" + dataSet['statement'].astype(str)

    ########################################################################################################################
    # New approach starting here
    ########################################################################################################################

    # HYPER PARAMETER

    A_slash, fre_hashTable = split_list_into_chunks(dataSet['template'], K)
    A_slash_slash = split_list_into_chunks(A_slash, G, False)


    # analyze_top_keys_variation(A_slash_slash, 100)

    """ 
    we now extract the tranning test from A_slash_slash

    The output is a 3D list -> [ [   [A single vector where length = TOP_RANK]   ] <- A single dataset for RNN where length = G       ]
    """
    # if isASmallTest:
        # return create_NN_input(A_slash_slash[:2000],TOP_RANK)
    # else:
    #     return create_NN_input(A_slash_slash,TOP_RANK)

    # return create_NN_input(A_slash_slash[:2000],TOP_RANK)


    # return create_NN_input(A_slash_slash,TOP_RANK)
    return create_NN_input_with_constant_TOP_RANK(A_slash_slash,TOP_RANK)



def nn_setup(sequence_list):

    feature_sequences=[]
    label_sequence=[]
    for subList in sequence_list:
        feature_sequences.append(subList[:len(subList)-1])
        label_sequence.append(subList[-1])
    return feature_sequences, label_sequence
    
            
