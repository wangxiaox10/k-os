# -*- coding: utf-8 -*-
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

#compares the precision of different alpha
def compareRes():
    outputFile_10 = "/home/xiao/test2/2/output2-11Jun-2"
    outputFile_20 = "/home/xiao/test2/1/output2-11Jun-1"
    
    outputFile_25 = "/home/xiao/test2/8/output2-11Jun-8"
    outputFile_30 = "/home/xiao/test2/6/output2-11Jun-6"
    
    
#    rank = numpy.genfromtxt(outputFile2, names=['Iteration', 'meanRank', 'maxRank', 'precisionAt10'])
    rank_10 = numpy.genfromtxt(outputFile_10, names=['Iteration', 'meanRank', 'maxRank','precisionAt1', 'precisionAt10'])
    precisionAt1List_10 = rank_10['precisionAt1']
    precisionAt10List_10 = rank_10['precisionAt10']
    
    rank_20 = numpy.genfromtxt(outputFile_20, names=['Iteration', 'meanRank', 'maxRank','precisionAt1', 'precisionAt10'])
    precisionAt1List_20 = rank_20['precisionAt1']
    precisionAt10List_20 = rank_20['precisionAt10']
    
    rank_25 = numpy.genfromtxt(outputFile_25, names=['Iteration', 'meanRank', 'maxRank','precisionAt1', 'precisionAt10'])
    precisionAt1List_25 = rank_25['precisionAt1']
    precisionAt10List_25 = rank_25['precisionAt10']
    
    rank_30 = numpy.genfromtxt(outputFile_30, names=['Iteration', 'meanRank', 'maxRank','precisionAt1', 'precisionAt10'])
    precisionAt1List_30 = rank_30['precisionAt1']
    precisionAt10List_30 = rank_30['precisionAt10']
    
    fig1 = plt.figure()
    
    plt.plot(precisionAt1List_10, 'y-', label="alpha = 1/(iteration/10+1)")
    plt.plot(precisionAt1List_20, 'b-', label="alpha = 1/(iteration/20+1)")
    plt.plot(precisionAt1List_25, 'g-', label="alpha = 1/(iteration/25+1)")
    plt.plot(precisionAt1List_30, 'r-', label="alpha = 1/(iteration/30+1)")
    
    plt.xlabel('iterations')
    plt.ylabel("precision@1")
    plt.legend(loc="lower right")
#    legend([precisionAt1List,precisionAt10List],["precision@1", "precision@10"])     
    
    plt.draw()
    
    fig2 = plt.figure()
    
    plt.plot(precisionAt10List_10, 'y-', label="alpha = 1/(iteration/10+1)")
    plt.plot(precisionAt10List_20, 'b-', label="alpha = 1/(iteration/20+1)")
    plt.plot(precisionAt10List_25, 'g-', label="alpha = 1/(iteration/25+1)")
    plt.plot(precisionAt10List_30, 'r-', label="alpha = 1/(iteration/30+1)")
    
    plt.xlabel('iterations')
    plt.ylabel("precision@10")
    plt.legend(loc="lower right")
#    legend([precisionAt1List,precisionAt10List],["precision@1", "precision@10"])     
    
    
compareRes()
