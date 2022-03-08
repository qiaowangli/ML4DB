#!/Users/royli/miniforge3/envs/tensorflow_m1/bin/python3

from matplotlib import pyplot as plt

class performance_visualizer():
    def __init__(self):
       self.plotting_list=[] # A list to store all the accuracy data.

    def data_append(self,data):
        self.plotting_list.append(data)
        
    def visual_plot(self):
        if len(self.plotting_list):
            plt.plot(self.plotting_list)
            plt.xlabel("splitting interval")
            plt.ylabel("performance")
            plt.show()
        else:
            print("the current length of plotting_list is 0")

    