# -*- coding: utf-8 -*-
# Author: Xiao.wang@polytechnique.edu
# Date: 24 mai, 2014

"""
 This files defines the overall parameters used in other files. 
 The users needs to give the correct value 
"""

# The complete path of file which stores the information of original observation data
#inputFile = "/home/xiao/ProjetLibre/ml-100/test.data"
inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
#inputFile = "/home/xiao/ProjetLibre/ml-100/test.data"
# The file which stores the output. Used to map data to graphs and other uses
outputFile = "output1"
outputFile2 = "output2"
#outputFile = "/home/xiao/remote_dir/rankRecommendation/output100k-alpha05"
#outputFile = "/home/xiao/remote_dir/rankRecommendation/output1-alpha05-2"
#dimension of user pattern
m = 64
#learning rate
alpha = 0.5


#order of element to pick. Used in preprocessData.py
i = 1
#number of elements to pick at each time. Explained in Algorithm1 in the paper. 
K = 15
#number of elements extracted from origin data for test
p = 5
#precision = 10. Decides when to stop the learning
precision = 0.01
#works with precision parameter
maxIteration = 100
#defines the number of loops the run before comparing current precsion with previous round 
iterationEachRound = 1

#Constraint C. C defines max norm of Vj
C = 1
