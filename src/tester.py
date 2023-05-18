from data_processor import raw_data_processor, generate_training_data, nn_setup
# from matplotlib import pyplot as plt
from performance_visualizer import performance_visualizer
from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
from vector2vector import word_embedding

from forcastor import rnn_regression
from cluster import kmean_cluster, Dbscan_cluster
import numpy as np
import os
from collections import defaultdict

import matplotlib.pyplot as plt

np.random.seed(0)




def main():


    # plt.figure(figsize=(8, 4))

    # YValue = [1184, 1210, 1402, 1625, 1961, 3046, 3674]
    # YValue_sub = [463, 469, 535, 573, 624, 801, 926]
    # YValue_sub1 = [128, 144, 169, 202, 289, 337, 400]
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.bar(range(len(YValue)), YValue, color='orange', hatch='o', edgecolor="black", label='Number of observation')
    # plt.bar(range(len(YValue_sub)), YValue_sub, color='pink', hatch='/', edgecolor="black", label='Number of invaild vector')
    # plt.bar(range(len(YValue_sub1)), YValue_sub1, color='green', hatch='@', edgecolor="black", label='Number of center')
    # plt.xlabel('Center reconstruction')
    # plt.ylabel('Number of observation')
    # plt.xticks(range(len(YValue)), [str(i) for i in range(len(YValue))], fontsize=3)
    # plt.legend()
    # plt.savefig('/Users/royli/Desktop/scatter_plot3.png', dpi=300)
    # plt.show()

    # exit(-1)
    """
    Version: V0.6

    """

    template_storage = {}
    sequence_storage = {}


    """ For pgbench dataset"""
    template_storage,sequence_storage=raw_data_processor("/Users/royli/Desktop/mldb/inputLogClear.csv",template_storage,sequence_storage,'query',6)
    sequenceList, NN_Input_Center= generate_training_data(sequence_storage, 140, 0.001,50)

    feature_sequences, label_sequence = nn_setup(sequenceList, 4)
    def create_frequency_hash_table(lst):
        frequency_table = defaultdict(int)
        for sublist in lst:
            frequency_table[tuple(sublist)] += 1
        return dict(frequency_table)
    testDic = create_frequency_hash_table(label_sequence)

    # print(max(testDic, key=testDic.get))

    keysT = list(testDic.keys())
    values = list(testDic.values())


    plt.figure(figsize=(12, 2))
    plt.bar(range(len(keysT)), values, color='green', hatch='*', edgecolor = "black")
    plt.xlabel('Categories')
    plt.ylabel('Frequencies')
    plt.title('Dictionary Plot')
    plt.xticks(range(len(keysT)), [str(i) for i in range(len(range(len(keysT))))], fontsize = 3)
    plt.savefig('/Users/royli/Desktop/scatter_plotX.png', dpi=300)


    


    print(len(feature_sequences[0][0]))
    # print(rnn_regression(feature_sequences, label_sequence,NN_Input_Center))



if __name__ == "__main__":
    main()
    