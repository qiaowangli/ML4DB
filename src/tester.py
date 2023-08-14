from data_processor import raw_data_processor , nn_setup
from forcastor import rnn_regression
import numpy as np
import matplotlib.pyplot as plt
import csv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

np.random.seed(0)



def plot_label_sequence(label_sequence):
    num_positions = len(label_sequence[0])

    plt.figure(figsize=(8, 6))

    for position in range(num_positions):
        values_at_position = [line[position] for line in label_sequence]
        plt.plot(range(1, len(label_sequence) + 1), values_at_position, label=f'Position {position+1}')

    with open('label_sequence_dataBig.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for position in range(num_positions):
            values_at_position = [line[position] for line in label_sequence]
            csv_writer.writerow(['Position {}'.format(position+1)] + values_at_position)

    plt.xlabel('Line')
    plt.ylabel('Value')
    plt.title('Label Sequence')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid(True)
    plt.show()


def main():

    """
    Version: V0.7

    """

    """ The number of template in a single time duration"""
    K = 1000

    """ The number of K in a single RNN forcasting """
    G = 4

    """ The number of ranked Sequences [The vertical input of RNN] """
    TOP_RANK = 10

    NN_input_3D_list=raw_data_processor("/home/ubuntu/ML4DB/inputLogClear.csv", K, G, TOP_RANK, False)
    feature_sequences, label_sequence = nn_setup(NN_input_3D_list)

    # print(label_sequence)
    # label_sequence = [[1, 2], [2, 4]]
    # plot_label_sequence(label_sequence)

    rnn_regression(feature_sequences, label_sequence)


  

if __name__ == "__main__":
    main()
    
