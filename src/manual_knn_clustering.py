#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from math import sqrt
import pandas as pd
import numpy as np
import random

"""
This file is for manual test only, for our model, we invoke the sk-learn API
"""
class knn_clustering():
  def __init__(self,dataset,k=10):
       self.k=k
       self.dataset=dataset
       
  def ramdon_setup(self,dataset):
    K=self.k
    #Our first stop would be ramdonly pick k centers and the range would be 0~5000
    init_centers = random.sample(range(0, len(dataset)),K)
    centers = []
    for i in init_centers:
      centers.append(dataset[i])
      #now we get k initial center
    return centers

  def Euclidean_distance(self,target, center):
      return sqrt((sum((target - center)**2)))
  #we use Euclidean distance to detect the nearest center 

  def pca(self,dataset):
    # now we call the sklearn library to train and fit.
    pca = PCA(0.95) # we want 95% variance to be explained
    return pca.fit_transform(dataset)


  def assign_center(self,dataset, centers):
    Nearest_center=[]
    for index in dataset:
      dis_center=[] #we use this array to store the distance to each center and than we pick the index of min number
      for center in centers:
        dis_center.append(self.Euclidean_distance(index,center))
      Nearest_center.append(np.argmin(dis_center))
    return Nearest_center

  def update_center_location(self,all_points, Nearest_center,centers):
    center_df=pd.DataFrame(data=Nearest_center,columns=['center'])
    data_main=pd.DataFrame(data=all_points)
    df_total=pd.concat([data_main,center_df],axis=1)

    for index in range(self.k):
      df_new=df_total[df_total["center"]==index]
      new_center=df_new.drop(['center'], axis=1).sum()/len(df_new)
      centers[index]=new_center # update the center location
    return np.array(centers)

  def K_mean_process(self,dataset,K_mean_pp_flag=False, iteration=10):
    dataset=np.array(dataset)
    if True:
      if K_mean_pp_flag:
        pass
      else:
        centers=self.ramdon_setup(dataset)

      # previous_center=[]
      # previous_center=np.array(previous_center)
      for iteraction in range(iteration):
        get_centroids = self.assign_center(dataset,centers)
        centers= self.update_center_location(dataset,get_centroids,centers)
        # if previous_center != [] and self.Euclidean_distance(previous_center,centers) < 10:
        #   break
        # previous_center=centers

    return self.assign_center(dataset,centers)
    
  def sequence_tokenization(self,centers,center_sequence):
    centers_dic={}
    for index in range(len(centers)): 
      centers_dic[centers[index]] = index 
    for sequence_index in range(len(center_sequence)):
      center_sequence[sequence_index]=centers_dic[center_sequence[sequence_index]]
    return center_sequence





      

