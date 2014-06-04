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
    data = numpy.genfromtxt(outputFile, names=['loss'])    
    loss = data['loss']
    
    fig1 = plt.figure()
    plt.plot(loss)
    plt.draw()
    
    fig2 = plt.figure()
    rank = numpy.genfromtxt(outputFile2, names=['meanRank', 'maxRank'])
    meanRankList = rank['meanRank']
    maxRankList = rank['maxRank']
    plt.plot( meanRankList, 'r-')
    plt.plot( maxRankList, 'g-')
    plt.draw()
    
showOutput()