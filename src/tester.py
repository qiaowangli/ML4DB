from data_processor import raw_data_processor, generate_training_data
# from matplotlib import pyplot as plt
from performance_visualizer import performance_visualizer
from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
from vector2vector import word_embedding

from forcastor import rnn_regression
from cluster import kmean_cluster, Dbscan_cluster
import numpy as np
import os
np.random.seed(0)




def main():
    """
    Version: V0.6

    """

    template_storage = {}
    sequence_storage = {}


    """ For pgbench dataset"""
    template_storage,sequence_storage=raw_data_processor("/Users/royli/Desktop/mldb/inputLogClear.csv",template_storage,sequence_storage,'query',6)
    sequenceList = generate_training_data(sequence_storage, 100, 0.001,50)
    # print(len(sequence_storage))
    print(len(sequenceList))
    print(len(sequenceList[2]))
    print(len(sequenceList[23]))
    # feature_sequences, label_sequence, sequence_storage= nn_setup(sequence_storage, 5)

    # print(len(centerList[0]))
    # print(rnn_regression(feature_sequences, label_sequence,centerList))



if __name__ == "__main__":
    main()
    