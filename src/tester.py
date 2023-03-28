
from data_processor import raw_data_processor, nn_setup, generate_training_data
# from matplotlib import pyplot as plt
from performance_visualizer import performance_visualizer
from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
from vector2vector import word_embedding

from forcastor import lstm_predictor,manual_test,rnn_classification, rnn_regression
from cluster import kmean_cluster, Dbscan_cluster
import numpy as np
import os
np.random.seed(0)




def main():
    """
    Version: V0.6 / rebuild the code for pytorch

    """

    template_storage = {}
    sequence_storage = {}


    """ For pgbench dataset"""
    template_storage,sequence_storage=raw_data_processor("/Users/royli/Desktop/ML4DB/inputLog.csv",template_storage,sequence_storage,'query',6)
    feature_sequences, label_sequence, sequence_storage= nn_setup(sequence_storage, 4)
    feature_sequences, label_sequence = generate_training_data(feature_sequences, label_sequence, sequence_storage)
    print(rnn_regression(feature_sequences, label_sequence))



if __name__ == "__main__":
    main()
    