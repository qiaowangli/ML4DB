#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3
import re
import time
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
            """
            The if statement checks if this is a valid query template we want.
            """
            new_template=re.sub('\d+', r'$$$', template_query[0][1])
            new_template=re.sub("\s+", r' ', new_template)
            new_template=re.sub("'(.*)'", r"'$$$'", new_template) 
            # concate duration level and query template
            new_template = tier_level + new_template
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


def nn_setup(sequence_list, time_step=10):
    """
    time_step indicates the step wise of RNN, if time_step is 1, the function generates the dataset for FNN.
    """
    feature_sequences=[]
    label_sequence=[]
    ending_point=0
    while ending_point+time_step <= len(sequence_list)-1:
        feature_sequences.append(sequence_list[ending_point:ending_point+time_step]) if time_step!=1 else feature_sequences.append(sequence_list[ending_point])
        label_sequence.append(sequence_list[ending_point+time_step])
        ending_point+=1
    return feature_sequences, label_sequence

