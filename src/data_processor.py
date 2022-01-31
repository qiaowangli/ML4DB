#!/usr/bin/python3
import re
from sys import stderr
import time
from datetime import datetime




##################################################################################
# These functions are designed to handle pgbench scenarios, if the number of #
# distinct query templates falls below 200, clustering will not be activated.    #
# Instead, One-to-hot would be used.                                             #
##################################################################################
def raw_data_processor(log_path,splitting_mode,predict_interval=5):
    """
    @input parameters:  log_path         -> log_file path, 
                        predict_interval -> furture interval to predict
                        splitting_mode   -> splitting_mode decides the way to split the queries

    @output: template_storage -> A dictionary that contains all distint queries and their distint ID.
             sequence_storage -> A dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    """
    log_file=open(log_path,"r")
    template_storage={} # dictionary contains all distint queries and their distint ID.
    sequence_storage={} # dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    template_index=0 
    if splitting_mode =="time":
        # we first get the starting date-time of the log file
        tracking_timestamp=int(time.mktime(datetime.strptime(log_file.readline().split('LOG:')[0].split(' PST')[0], "%Y-%m-%d %H:%M:%S").timetuple()))
        # we create the dic for the first timestamp
        sequence_storage[tracking_timestamp]={}
    elif splitting_mode =="query":
        query_group_index=0
        sequence_storage[query_group_index]={}
    else:
        print("Invalid splitting_mode, the system only supports 'time' and 'query' feature")
        exit(-1)
    
    for line in log_file:
        try:
            query_timestamp=int(time.mktime(datetime.strptime(line.split('LOG:')[0].split(' PST')[0], "%Y-%m-%d %H:%M:%S").timetuple()))
            template_query=re.findall(r"^duration:(.*)ms. statement: (.*)\n",line.split(' LOG:  ')[1])
        except:
            continue

        if splitting_mode == "time" and query_timestamp not in sequence_storage and query_timestamp > tracking_timestamp+predict_interval:
            sequence_storage[query_timestamp]={}
            tracking_timestamp=query_timestamp
        elif splitting_mode == "query" and sum(sequence_storage[query_group_index].values()) > predict_interval:
            query_group_index+=1
            sequence_storage[query_group_index]={} # create a new dictionary


        if template_query:
            new_template=re.sub('\d+', r'$$$', template_query[0][1]) 
            
            # check if this is a new template
            if new_template not in template_storage:
                template_storage[new_template]=template_index
                template_index+=1
            # the variable 'interval_id' is used to identify which sub_dictionary the system is working with.
            interval_id=tracking_timestamp if splitting_mode == "time" else query_group_index

            #record the queries 
            if template_storage[new_template] not in sequence_storage[interval_id]:
                sequence_storage[interval_id][template_storage[new_template]]=1
            else:
                sequence_storage[interval_id][template_storage[new_template]]+=1


    return template_storage,sequence_storage

def sequence_producer(templates_num,sequence_storage):
    """
    This function aims to convert the a dictonary to a set of sequences, then we can think of it as a time series problem

    @input parameters : templates_num    -> the total numbers of distint template.
                        sequence_storage -> A dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    
    @output: sequence_list -> a set of sequences for time series problem.
    """
    sequence_list=[]
    for timestamp in sorted(sequence_storage):
        sub_sequence=[0]*templates_num
        if sequence_storage[timestamp]:
            for id in sequence_storage[timestamp]:
                sub_sequence[id]=sequence_storage[timestamp][id]
        sequence_list.append(sub_sequence)
    return sequence_list

def nn_setup(sequence_list, time_step=10):
    feature_sequences=[]
    label_sequence=[]
    ending_point=0
    while ending_point+time_step <= len(sequence_list)-1:
        feature_sequences.append(sequence_list[ending_point:ending_point+time_step])
        label_sequence.append(sequence_list[ending_point+time_step])
        ending_point+=1
    return feature_sequences, label_sequence

