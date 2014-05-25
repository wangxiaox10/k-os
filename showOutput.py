# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 25 Mai, 2014
import config
import matplotlib.pyplot as plt
import numpy

def showOutput():
    outputFile = config.outputFile
    data = numpy.genfromtxt(outputFile, names=['iteration','loss'])    
    iteration = data['iteration']
    loss = data['loss']
    
    print iteration
    print loss
    
    fig1 = plt.figure()
    plt.scatter(iteration,loss)
    plt.draw()

showOutput()