#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3
import re
import time
import numpy as np
from datetime import datetime


""" GLOBAL VARIABLE """


def raw_data_processor(log_path,template_storage,sequence_storage,template_index,duration_time,splitting_mode="time",predict_interval=5):
    """
    @input parameters:  log_path         -> log_file path
                        splitting_mode   -> splitting_mode decides the way to split the queries
                        predict_interval -> furture interval to predict
                        
    @output: template_storage -> A dictionary that contains all distint queries and their distint ID.
             sequence_storage -> A dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
    """
    vaild_query_type=['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']

    log_file=open(log_path,"r")
    if not template_storage and not sequence_storage:
        pass
        # template_storage={} # dictionary contains all distint queries and their distint ID.
        # sequence_storage={} # dictionary contains a set of timestamps, where each timestamp records the frequency of query IDs executed during the timestamp period.
        # template_index=0 
    if splitting_mode =="time":
        # we first get the starting date-time of the log file
        """
        Below command is for pgbench data processing
        """
        tracking_timestamp=float(log_file.readline().split('LOG:')[0].split(' ')[0])
        """
        Below command is for github data processing
        """
        # try:
        #     tracking_timestamp=int(datetime.strptime(log_file.readline().split('statement')[0].split(' EST')[0], "\"%Y-%m-%d %H:%M:%S.%f").timestamp()*1000)
        # except:
        #     print("invaild file, skip it")
        #     exit()

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
            """
            Below command is for pgbench data processing
            """
            query_timestamp=float(line.split('LOG:')[0].split(' ')[0])
            template_query=re.findall(r"^duration:(.*)ms. statement: (.*)\n",line.split(' LOG:  ')[1])
            """
            Below command is for github data processing
            """
            # query_timestamp=int(datetime.strptime(line.split('statement')[0].split(' EST')[0], "\"%Y-%m-%d %H:%M:%S.%f").timestamp()*1000)
            # template_query=re.findall(r"\"statement: (.*) \"",line)
        except:
            continue

        if splitting_mode == "time" and query_timestamp not in sequence_storage and query_timestamp > tracking_timestamp+predict_interval:
            sequence_storage[query_timestamp]={}
            tracking_timestamp=query_timestamp
        elif splitting_mode == "query" and sum(sequence_storage[query_group_index].values()) > predict_interval:
            query_group_index+=1
            sequence_storage[query_group_index]={} # create a new dictionary

        """
        The if statement should change to 
        " if template_query and template_query[0].split(" ")[0] in vaild_query_type: "
        with github dataset
        """
        if template_query and template_query[0][1].split(" ")[0] in vaild_query_type:

            duration_number = float( template_query[0][0])
            duration_time.append(duration_number)
            tier_level=None 
            if duration_number > 0.1:
                tier_level = "tier_10_"
            else:
                tier_level= "tier_"+str(int(duration_number*100)%10)+"_"
            """ The following code is used to test the effect of the length of the input vector on the accuracy """
            # if duration_number > 0.6:
            #     tier_level = "tier_0_"
            # elif duration_number > 0.3:
            #     tier_level = "tier_1_"
            # else:
            #     tier_level = "tier_2_"

            """
            The if statement checks if this is a valid query template we want.
            """
            new_template=re.sub('\d+', r'$$$', template_query[0][1])
            new_template=re.sub("\s+", r' ', new_template)
            new_template=re.sub("'(.*)'", r"'$$$'", new_template) 
            # concate duration level and query template
            new_template = tier_level + new_template

            # print(new_template)
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


    return template_storage,sequence_storage,template_index,duration_time

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


def nn_setup(sequence_list,time_step=10,skipgram=False,new_approach=False):
    """
    time_step indicates the step wise of RNN, if time_step is 1, the function generates the dataset for FNN.
    """
    feature_sequences=[]
    label_sequence=[]
    total_sequence=[]
    ending_point=0
    if not skipgram:  
        while ending_point+time_step <= len(sequence_list)-1:
            feature_sequences.append(sequence_list[ending_point:ending_point+time_step]) if time_step!=1 else feature_sequences.append(sequence_list[ending_point])
            if not new_approach:
                label_sequence.append(sequence_list[ending_point+time_step])
            else:
                label_sequence.append(ending_point+time_step)# we just record the index for further usage
            total_sequence.append(sequence_list[ending_point:ending_point+time_step+1])
            ending_point+=1

    return feature_sequences, label_sequence,total_sequence

def generate_training_data(feature_sequences, label_sequences,embedding_table,tokenized_sequence_list_target,classification_task=False, new_approach=False):
    # print(type(embedding_table))
    embedding_feature_sequences=[]
    embedding_label_sequences=[]
    for index in range(len(label_sequences)):
        if not classification_task:
            embedding_label_sequences.append(embedding_table[label_sequences[index]])
        else:
            if new_approach:
                label_layer_length=len(set(tokenized_sequence_list_target))
                # print(tokenized_sequence_list_target[label_sequences[index]])
                # print(label_layer_length)
                embedding_label_sequences.append(np.eye(label_layer_length)[tokenized_sequence_list_target[label_sequences[index]]])
            else:
                embedding_label_sequences.append(np.eye(len(embedding_table))[label_sequences[index]])
        sub_list=[]
        for senario in feature_sequences[index]:
            sub_list.append(embedding_table[senario])
        embedding_feature_sequences.append(sub_list)
    return embedding_feature_sequences,embedding_label_sequences



def tokenization(sequence_list):
    """
    input : sequence_list -> [[s1],[s2],[s3],[s4] ....,[sn]] where sn is prediction vector
    output: tokenized sequence_list -> [1,2,3,2,3,....,678] and a look_up table -> {[s1]:0,[s2]:1,[s3]:2,[s4]:3 ....,[sn]:n-1}
    """
    lookup_table={}
    tokenized_sequence_list=[]
    index=0
    for predicton_vector in sequence_list:
        if predicton_vector not in lookup_table.values():
            lookup_table[index]=predicton_vector
            tokenized_sequence_list.append(index)
            index+=1
        else:
            # each key must have a unique value
            tokenized_sequence_list.append(list(lookup_table.keys())[list(lookup_table.values()).index(predicton_vector)])
    return tokenized_sequence_list,lookup_table


