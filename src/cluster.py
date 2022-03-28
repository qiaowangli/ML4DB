#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class kmean_cluster():
    def __init__(self,dataset,k=10):
       self.k=k
       self.dataset=dataset

    def pca(self,dataset):
        # now we call the sklearn library to train and fit.
        pca = PCA(0.95) # we want 95% variance to be explained
        self.dataset=pca.fit_transform(dataset)

    def kmean(self):
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.dataset)
        return kmeans
    
    def classification(self,model):
        # return model.predict(self.dataset)
        return model.labels_


class Dbscan_cluster():
    def __init__(self,dataset,epsilon=1):
       self.epsilon=epsilon
       self.dataset=dataset

    def pca(self,dataset):
        # now we call the sklearn library to train and fit.
        pca = PCA(0.95) # we want 95% variance to be explained
        self.dataset=pca.fit_transform(dataset)

    def Dbscan(self):
        Dbscan_model = DBSCAN(eps=self.epsilon, min_samples=2).fit(self.dataset)
        return Dbscan_model
    
    def classification(self,model):
        return model.labels_