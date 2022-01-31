#!/usr/bin/python3
from data_processor import raw_data_processor,sequence_producer,nn_setup
from performance_visualizer import performance_visualizer
from matplotlib import pyplot as plt
from lstm import lstm_predictor
import numpy as np
np.random.seed(0)


def main():
    """
    The current version(v0.2) supports 2 splitting modes which are 'time' and 'query', the default parameter is 'time'
    The variable index_interval implies the time duration of a bucket when splitting modes parameter is 'time',
    otherwise it means the maximum amount of query storage for a single bucket
    """
    plotter=performance_visualizer() # create an plotting object
    for index_interval in range(1,40,3):
      template_storage,sequence_storage=raw_data_processor('/content/postgresql-1.log',index_interval,'time')
      sequence_list=sequence_producer(len(template_storage),sequence_storage)
      feature_sequences, label_sequence=nn_setup(sequence_list)
      plotter.data_append(lstm_predictor(feature_sequences, label_sequence)[1])
    
    plotter.plotting_list()





if __name__ == "__main__":
    main()
    