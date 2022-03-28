#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3

from pickle import TRUE
from data_processor import raw_data_processor,sequence_producer,nn_setup,tokenization,generate_training_data
from matplotlib import pyplot as plt
from performance_visualizer import performance_visualizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from forcastor import lstm_predictor,manual_test,rnn_classification
from vector2vector import word_embedding
from cluster import kmean_cluster, Dbscan_cluster
import numpy as np
import os
np.random.seed(0)




def main():
    """
    Version: V0.5
    # The following are hyperparameters that can be used for tuning

    step 1(build sequence) : splitting interval[float]

    step 2(clustering)     : k(kmean)[int]/ epsilon(DBSCAN)[float]

    step3(time_step)       : time_step[int]

    step4(word_embedding)  : window_size[int], negative_sampling[int], embedding_dim[int]

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
    template_storage,sequence_storage,template_index,duration_time=raw_data_processor("/Users/royli/Desktop/csc490/postgresql-10_2400_ms.log",template_storage,sequence_storage,template_index,duration_time,'time',0.1)
    sequence_list=sequence_producer(len(template_storage),sequence_storage)
    """ Kmean++ builder """
    test=kmean_cluster(sequence_list,k=6)
    test.pca(sequence_list)
    kmeans_model=test.kmean()
    tokenized_sequence_list = test.classification(kmeans_model)
    distint_value_length = len(kmeans_model.cluster_centers_)
    """ DBSCAN builder """
    # test=Dbscan_cluster(sequence_list,epsilon=1)
    # test.pca(sequence_list)
    # dbscan_model=test.Dbscan()
    # tokenized_sequence_list = test.classification(dbscan_model)
    # tokenized_sequence_list=np.array(tokenized_sequence_list)
    # distint_value_length=len(np.unique(tokenized_sequence_list))
     # for index in range(len(tokenized_sequence_list)):
        #     if tokenized_sequence_list[index] == -1:
        #         tokenized_sequence_list[index]=distint_value_length
    """ Til this step, we should get a tokenization_list where each integer in this list is the index of the center list of Kmean++ """

    feature_sequences, label_sequences,total_sequence=nn_setup(tokenized_sequence_list,time_step=100) # the dataset is for FNN if time_step is 1
    print(total_sequence[25])
    print(feature_sequences[25])
    print(label_sequences[25])
    print(distint_value_length)
    embedding_table = word_embedding(total_sequence,distint_value_length,window_size=5,ns_size=5)
    embedding_feature_sequences, embedding_label_sequences = generate_training_data(feature_sequences,label_sequences,embedding_table,classification_task=False)
    print(lstm_predictor(embedding_feature_sequences, embedding_label_sequences,baseline_test=False))





if __name__ == "__main__":
    main()
    