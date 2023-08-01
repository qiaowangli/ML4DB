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



INITIAL_TEMPLATE_OBSERVATION = -1

def split_list_into_chunks(A, k, stepK = True):
    if k <= 0 or not isinstance(k, int):
        raise ValueError("k must be a positive integer")

    if len(A) < k:
        raise ValueError("k must be less than or equal to the length of the input list")
    
    if stepK:
        A = A.tolist()  # Convert the Pandas Series to a Python list

    result = [A[i:i + k] for i in range(len(A) - k + 1)]
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
    plt.ylabel('Top 5 Keys Variation Count')
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

    """ The number of template in a single time duration"""
    K = 100

    """ The number of K in a single RNN forcasting """
    G = 20

    """ The number of ranked Sequences [The vertical input of RNN] """
    TOP_RANK = 10

    A_slash = split_list_into_chunks(dataSet['template'], K)
    A_slash_slash = split_list_into_chunks(A_slash, G, False)

    # analyze_top_keys_variation(A_slash_slash, 100)

    """ 
    we now extract the tranning test from A_slash_slash

    The output is a 3D list -> [ [   [A single vector where length = TOP_RANK]   ] <- A single dataset for RNN where length = G       ]
    """
    return create_NN_input(A_slash_slash[:2000],TOP_RANK)



def nn_setup(sequence_list):

    feature_sequences=[]
    label_sequence=[]
    for subList in sequence_list:
        feature_sequences.append(subList[:len(subList)-1])
        label_sequence.append(subList[-1])
    return feature_sequences, label_sequence
    
            
