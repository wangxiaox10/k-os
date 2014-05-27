# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date:   22 May, 2014

"""
This file implements the Algorithm3 in the paper. 
Being tested in file testAUC.py
the input is obversation matrix X, test matrix test, m which defines the dimension of matrix V
the output is V after learning processus 
"""
from preprocessData import * 
import numpy
import config

def k_os_AUC_loss(X, m, test):

    """
    0. constants
    """
    #n number of users
    n = X.shape[1]


    """
    1. initialise the model
    """
    # matrix of approximation to learn
    V = model(m,n).getV()
    
    
    """
    2. repeat
    until the error does not improve
    """
    # object lossFunction to compute the loss etc. 
    lossFunc = lossFunction()
    # object numericalInterest to compute how a user likes items d
    nInterest = numericalInterest()
    
    # initial loss before the loop
    previousLoss = lossFunc.AUCLoss(X,V)
    
    currentLoss = -1
    countIteration = 0
    
    while True:
        
        countIteration+=1
        """
        check first iteration? 
        """
        if currentLoss != -1:
            previousLoss = currentLoss

        """
        show result on terminal and stores in output file
        """
        
        print countIteration, previousLoss
        resToWrite = str(countIteration) + " " + str(previousLoss)+"\n"
        f_output = open(config.outputFile, 'a')
        f_output.write(resToWrite)
        f_output.close()
        
        lossSum = 0
        iterationRoundCount = 0
        for i in range(config.iterationEachRound):
            """
            pick a positive item d using algorithm 1
            """
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
                alpha = config.alpha
                
                V = lossFunc.SGD(X, V,u, d, bar_d, alpha)
                """
                Project weights to enforce constraints:
                ensure ||Vi|| <= C
                """
                C = config.C
                V = lossFunc.constraintNorm(V, C)
                
                currentLoss = lossFunc.AUCLoss(X,V)
                lossSum += currentLoss
                iterationRoundCount += 1
                
        if( iterationRoundCount > 0):
            currentLoss = (lossSum + 0.0) / iterationRoundCount
            predictRanking(test, config.p, X, V)

            if (countIteration > config.maxIteration):
                """
                if validation error doesn't improve
                stop the loop
                """
                if (numpy.abs(currentLoss - previousLoss)<config.precision):
                    f_output = open(config.outputFile, 'a')
                    resToWrite = "#finish learning," + str(countIteration) + "\n#total loss:" + str(currentLoss)+"\n"
                    f_output.write(resToWrite)
                    f_output.close()
                    print "#finish learning", countIteration
                    print "#total loss:", currentLoss
                    break
        
    return  V