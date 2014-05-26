# -*- coding: utf-8 -*-
# Author : xiao.wang@polytechnique.edu
# Date : 26 Mai, 2014

import numpy
import config
import lossFunction

def k_os_WARP_loss(X, m):
    #lists that stores (iteration, loss at an iteration)
    axisIteration=[]
    axisLoss=[]
    
    lossFunc = lossFunction.lossFunction()
    """
    1. initialise the model
    """
#    X = preprocessData().getX()
    n = X.shape[1]
    V = model(m,n).getV()
    
    countIteration = 0
    previousLoss = lossFunc.WARPLoss(X,V)
    while True:
        """
        a. pick a random item d using algorithm 1
        """
        (u, d) = pickingPositiveItem(X,V)
        N = 0
        
        """
        b. pick a bar_d at random from D\Du
        """
        while True:
            
            bar_Du = lossFunc.getBarDu(X, u)
            bar_d = numpy.random.choice(bar_Du)
            
            N += 1
            
            f_d_u = nInterest.f_d(d,u,X,V)
            f_bar_d_u = nInterest.f_d(bar_d, u, X, V)
    
            """
            until f_bar_d_u > f_d_u - 1 or N >= #(D-Du)
            """
            if ((f_bar_d_u > f_d_u-1 ) or(N >= n-len(Du))):
                break
            
        if(f_bar_d_u > f_d_u-1):
            countIteration += 1
            """
            1. make a gradient step to minimize
            """
            alpha = config.alpha
            V = lossFunc.SGD(X, V,u, d, bar_d, alpha)
                
            """
            2. Project weights to enforce constrains
            """
            C = config.C
            V = lossFunc.constraintNorm(V, C)
               
            """
            3. compute validation error
            """
            if(countIteration > 1):
                previousLoss = currentLoss
            currentLoss = lossFunc.WARPLoss(X, V)
            
            
        """
        if validation error doesn't improve, stop the loop
        """
        if (numpy.abs(currentLoss - previousLoss)<config.precision):
            f_output = open(config.outputFile, 'a')
            resToWrite = "#finish learning," + str(countIteration) + "\n#total loss:" + str(currentLoss)+"\n"
            f_output.write(resToWrite)
            f_output.close()
            print "#finish learning", countIteration
            print "#total loss:", currentLoss
            break
