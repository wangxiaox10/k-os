# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 24 Mai, 2014

"""
This file tests the correctness of the learning result. 
After learning the matrix V, we test it with matrix test, which is created in the preprocessData.py
For each user u, we generate the p most liked items to recommend, according to f_d(numericalInterest.py)
and V(generated from K_AUC_Loss.py or K_WARP_Loss.py), and compare recommendations with test values. 
We accumulate errors and show error rate on terminal. 
The lower the error rate, the better the learning processus is. 
"""

from lossFunction import * 
from pickingPositiveItem import *
from numericalInterest import *
from K_AUC_Loss import *
import numpy
import config

def predictRanking(test, p, X, V):
    """
    0. constants
    """
    nUser = X.shape[0]
    nInterest = numericalInterest()
    lossFunc = lossFunction()
    
    """
    1. initialisation
    """
    error = 0
    
    """
    2. iterate each user
    """
    for u in range(nUser):
        
        # bar_Du: set of unrated items for user u, including test items
        bar_Du = lossFunc.getBarDu(X, u)
        
        # interest rates for these items 
        f_u_rating_of_unknown_items = nInterest.f_set(bar_Du, u, X, V)
        # order the items in descending order on corresponding value
        # f_order stores the index of items in bar_Du
        f_order = numpy.argsort(f_u_rating_of_unknown_items)
        f_order = f_order[::-1] 
        
        '''
        recommend 2*p items 
        '''
        recommend = bar_Du[f_order][:2*p]
        
        for i in test[u]:
            if i not in recommend:
                error += 1
    
    print "#error:", error, "percentage: ", (error+0.0)/(nUser * p)
    resStr = "#error:"+ str(error)+", percentage: "+str( (error+0.0)/(nUser * p))
    f_output = open(config.outputFile, 'a')
    f_output.write(resStr)
    f_output.close()
