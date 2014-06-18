# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 25 Mai, 2014

"""
This files allows the show the result on graph. 
The result is stored in a file, either on local computer or on remote cluster. 
For every line in the file, the data is stored in the form
iteration_time loss\n
"""

import config
import matplotlib.pyplot as plt
import numpy

def showOutput():
    outputFile = config.outputFile
    outputFile2 = config.outputFile2
    data = numpy.genfromtxt(outputFile, names=['Iteration', 'loss'])   
    Iteration = data['Iteration']
    loss = data['loss']

    #sorting data
    loss  = loss[Iteration.argsort()]
    
    fig1 = plt.figure()
    plt.plot(Iteration[Iteration.argsort()], loss, label="k-os loss")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.draw()
    
    fig2 = plt.figure()
#    rank = numpy.genfromtxt(outputFile2, names=['Iteration', 'meanRank', 'maxRank', 'precisionAt10'])
    rank = numpy.genfromtxt(outputFile2, names=['Iteration', 'meanRank', 'maxRank','precisionAt1', 'precisionAt10'])
    Iteration2 = rank['Iteration']
    meanRankList = rank['meanRank']
    maxRankList = rank['maxRank']
    precisionAt1List = rank['precisionAt1']
    precisionAt10List = rank['precisionAt10']
    
    meanRankList = meanRankList[Iteration2.argsort()]
    maxRankList = maxRankList[Iteration2.argsort()]
    
    plt.plot( meanRankList, 'r-',label="meanRank")
    plt.plot( maxRankList, 'g-',label="maxRank")
    plt.xlabel('iterations')
    plt.ylabel("rank")
    plt.legend(loc='center right')

#    legend([meanRankList, maxRankList], ["meanRankList", "maxRankList"])
    plt.draw()
    
    fig3 = plt.figure()
    plt.plot(precisionAt1List, 'y-', label="P@1")
    plt.plot(precisionAt10List, 'b-', label="P@10")
    plt.xlabel('iterations')
    plt.ylabel("precision")
    plt.legend(loc="lower right")
#    legend([precisionAt1List,precisionAt10List],["precision@1", "precision@10"])     
    plt.draw()
    
showOutput()