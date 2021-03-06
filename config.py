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
# The file which stores the output. Used to map data to graphs and other uses

#outputFile = "/home/xiao/test1/8/output1-100k-9Jun"
#outputFile2 = "/home/xiao/test1/8/output2-100k-9Jun"

outputFile = "/home/xiao/test2/4/output1-11Jun-4"
outputFile2 = "/home/xiao/test2/4/output2-11Jun-4"


#outputFile  = "output1-test"
#outputFile2 = "output2-test"
#
#outputFile = "/home/xiao/remote_dir/k-os/output1-11Jun-1"
#outputFile2 = "/home/xiao/remote_dir/k-os/output2-11Jun-1"

#outputFile = "/home/xiao/remote_dir/k-os/output1-100k-9Jun-2"
#outputFile2 = "/home/xiao/remote_dir/k-os/output2-100k-9Jun-2"


#outputFile = "/home/xiao/remote_dir/k-os/output1-test-100k-3"
#outputFile2 = "/home/xiao/remote_dir/k-os/output2-test-100k-3"

#dimension of user pattern
m = 64
#learning rate
alpha = 0.05


#order of element to pick. Used in preprocessData.py
i = 1
#number of elements to pick at each time. Explained in Algorithm1 in the paper. 
K = 5
#number of elements extracted from origin data for test
p = 5
#precision = 10. Decides when to stop the learning
precision = 0.01
#works with precision parameter
maxIteration = 80000
#defines the number of loops the run before comparing current precsion with previous round 
iterationEachRound = 1

#Constraint C. C defines max norm of Vj
C = 10
atPrecision = 10