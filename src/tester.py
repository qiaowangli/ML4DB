from data_processor import raw_data_processor , nn_setup
from forcastor import rnn_regression
import numpy as np

np.random.seed(0)




def main():

    """
    Version: V0.7

    """

    template_storage = {}
    sequence_storage = {}


    NN_input_3D_list=raw_data_processor("/Users/royli/Desktop/mldb/inputLogClear.csv",template_storage,sequence_storage,'query',6)
    feature_sequences, label_sequence = nn_setup(NN_input_3D_list)
    rnn_regression(feature_sequences, label_sequence)


  

if __name__ == "__main__":
    main()
    
