# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date:   22 May, 2014
from lossFunction import * 
from pickingPositiveItem import *
from numericalInterest import *
import numpy
import matplotlib.pyplot as plt
import config

def k_os_AUC_loss(X, m):
#    fig1 = plt.figure()
    axisIteration=[]
    axisLoss=[]
    
    """
    1. initialise the model
    """
#    X = preprocessData().getX()
#    m = 2  # learning dimension 
#    n = 4  # number of items 
    n = X.shape[1]
    V = model(m,n).getV()
    
    """
    2. repeat
    until the error does not improve
    """
#    epsilon  = 1000
    lossFunc = lossFunction()
    nInterest = numericalInterest()
    
    previousLoss = lossFunc.AUCLoss(X,V)
    currentLoss = -1
    countIteration = 0
    while True:
        
        countIteration+=1
        if currentLoss != -1:
            previousLoss = currentLoss
#        print "iteration:", countIteration, "loss:", previousLoss
        print countIteration, previousLoss
        resToWrite = str(countIteration) + " " + str(previousLoss)+"\n"
        f_output = open(config.outputFile, 'a')
        f_output.write(resToWrite)
        f_output.close()
        
        axisIteration.append(countIteration)
        axisLoss.append(previousLoss)
#        plt.scatter(axisIteration,axisLoss)
#        plt.draw()
#        plt.show()
        
        (u, d) = pickingPositiveItem(X,V)
        
        """
        pick a bar_d at random from D\Du
        """
        bar_Du = lossFunc.getBarDu(X, u)
        bar_d = numpy.random.choice(bar_Du)
        
        f_d_u = nInterest.f_d(d,u,X,V)
        f_bar_d_u = nInterest.f_d(bar_d, u, X, V)
        
        if (f_bar_d_u > f_d_u -1):
            """
            make a gradient step
            """
#            alpha = 0.5
            alpha = config.alpha
            
            V = lossFunc.SGD(X, V,u, d, bar_d, alpha)
            """
            Project weights to enforce constraints:
            ensure ||Vi|| <= C
            """
            C = 2
            V = lossFunc.constraintNorm(V, C)
        currentLoss = lossFunc.AUCLoss(X,V)

#        if (numpy.abs(currentLoss - previousLoss)<epsilon):
        if (countIteration > config.maxIteration):
            if (numpy.abs(currentLoss - previousLoss)<config.precision):
                f_output = open(config.outputFile, 'a')
                resToWrite = "#finish learning," + str(countIteration) + "\n#total loss:" + str(currentLoss)+"\n"
                f_output.write(resToWrite)
                f_output.close()
                print "#finish learning", countIteration
                print "#total loss:", currentLoss
                break
        
    return  V
#k_os_AUC_loss()