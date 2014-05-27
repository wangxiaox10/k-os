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
    data = numpy.genfromtxt(outputFile, names=['iteration','loss'])    
    iteration = data['iteration']
    loss = data['loss']
    
    fig1 = plt.figure()
    plt.plot(iteration,loss)
    plt.draw()
