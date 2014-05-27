# -*- coding: utf-8 -*-
# Author: Xiao.wang@polytechnique.edu
# Date: 24 mai, 2014

"""
 This files defines the overall parameters used in other files. 
 The users needs to give the correct value 
"""

# The complete path of file which stores the information of original observation data
inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
# The file which stores the output. Used to map data to graphs and other uses
outputFile = "output1"

#dimension of user pattern
m = 30
#learning rate
alpha = 0.01


#order of element to pick. Used in preprocessData.py
i = 1
#number of elements to pick at each time. Explained in Algorithm1 in the paper. 
K = 20
#number of elements extracted from origin data for test
p = 1
#precision = 10. Decides when to stop the learning
precision = 0.0001
#works with precision parameter
maxIteration = 800
#defines the number of loops the run before comparing current precsion with previous round 
iterationEachRound = 1

#Constraint C. C defines max norm of Vj
C = 2
