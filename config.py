# -*- coding: utf-8 -*-
#import datetime

inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
#outputFile = "/home/xiao/remote_dir/k-os/output1"
#outputFile = "/home/xiao/remote_dir/k-os/output1-alpha1"
#outputFile = "/home/xiao/remote_dir/k-os/output1-alpha0dot1"
#outputFile = "output1"
outputFile = "/home/xiao/remote_dir/k-os/output1-alpha5"
fileErrorRate = "output2"
#dimension of user pattern
m = 20
#learning speed 
alpha = 0.1
#order of element to pick 
i = 1
#number of elements to pick at each time
K = 20
# number of elements extracted from origin data for test
p = 5
#precision = 10
precision = 0.0001
maxIteration = 10000
iterationEachRound = 20

#Constraint C
C = 2
#min(lossAUC) = 1.109e7
