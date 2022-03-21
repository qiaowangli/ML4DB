#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3

from pickle import TRUE
from data_processor import raw_data_processor,sequence_producer,nn_setup,tokenization,generate_training_data
from matplotlib import pyplot as plt
from performance_visualizer import performance_visualizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from forcastor import lstm_predictor,manual_test
from vector2vector import word_embedding
import numpy as np
import os
np.random.seed(0)




def main():
    """
    The current version(v0.4) supports 2 splitting modes which are 'time' and 'query' and duration concatenation, the default parameter is 'time'
    The variable index_interval implies the time duration of a bucket when splitting modes parameter is 'time',
    otherwise it means the maximum amount of query storage for a single bucket

    The current version in main branch deals with pgbench dataset.
    The current version in dev branch deals with github dataset.
    """

    template_storage = {}
    sequence_storage = {}
    duration_time = []
    template_index = 0
    """ For github dataset"""
    # pathlist=os.listdir("/Users/royli/Desktop/tiramisu-sample")
    # pathlist.sort()
    # for filename in pathlist:
    #     if filename != '.DS_Store':
    #         template_storage,sequence_storage,template_index=raw_data_processor(os.path.join("/Users/royli/Desktop/tiramisu-sample",filename),template_storage,sequence_storage,template_index,'time',0.1)
    """ For pgbench dataset"""
    template_storage,sequence_storage,template_index,duration_time=raw_data_processor("/Users/royli/Desktop/csc490/postgresql-10_2400_ms.log",template_storage,sequence_storage,template_index,duration_time,'time',0.01)
    sequence_list=sequence_producer(len(template_storage),sequence_storage)
    tokenized_sequence_list,lookup_table=tokenization(sequence_list)
    feature_sequences, label_sequences,total_sequence=nn_setup(tokenized_sequence_list,time_step=10) # the dataset is for FNN if time_step is 1
    embedding_table = word_embedding(total_sequence, len(lookup_table))
    embedding_feature_sequences, embedding_label_sequences = generate_training_data(feature_sequences, label_sequences,embedding_table)
    print(lstm_predictor(embedding_feature_sequences, embedding_label_sequences,baseline_test=False))





if __name__ == "__main__":
    main()
    