#!/usr/bin/python3
from ML4DB.src.data_processor import raw_data_processor,sequence_producer,nn_setup
from ML4DB.src.lstm import lstm_predictor


def main():

    template_storage,sequence_storage=raw_data_processor('../data/pg_log/postgresql-4.log')
    sequence_list=sequence_producer(len(template_storage),sequence_storage)
    feature_sequences, label_sequence=nn_setup(sequence_list)
    lstm_predictor(feature_sequences, label_sequence)




if __name__ == "__main__":
    main()
    