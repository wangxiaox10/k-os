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

def predictRanking(test, p, X, V, numIteration):
    """
    0. constants
    """
#    print "test:", test
    nUser = X.shape[0]
    nInterest = numericalInterest()
    lossFunc = lossFunction()
    
    """
    1. initialisation
    """
    sumMeanRank = 0
    sumMaxRank = 0
    
    """
    2. iterate each user
    """
#    print "testing ..."
    
    for u in range(nUser):
        
        # bar_Du: set of unrated items for user u, including test items
        bar_Du = lossFunc.getBarDu(X, u)
        
        f_u_testing_items = nInterest.f_set(test[u].astype(int), u, X, V)
        # interest rates for these items 
        f_u_rating_of_unknown_items = nInterest.f_set(bar_Du, u, X, V)
        # order the items in descending order on corresponding value
        # f_order stores the index of items in bar_Du
#        f_order = numpy.argsort(f_u_rating_of_unknown_items)
        f_u_rating_of_unknown_items.sort()
#        f_u_rating_of_unknown_items = f_u_rating_of_unknown_items[::-1]
#        f_order = f_order[::-1] 
        
        """
        compute mean rank: average rank of tested items in the recommendation list
        compute max rank: largest rank of tested items in the recommendation list
        """
        
        testingItemRank = len(f_u_rating_of_unknown_items) - numpy.searchsorted(f_u_rating_of_unknown_items, f_u_testing_items)
        mean_rank = numpy.mean(testingItemRank)
        max_rank = numpy.max(testingItemRank)
        
        sumMeanRank += mean_rank
        sumMaxRank += max_rank
        
#        '''
#        recommend 2*p items 
#        '''
        
#        recommend = bar_Du[f_order][:2*p]
#        
#        for i in test[u]:
#            if i not in recommend:
#                error += 1
#        print test[u]
#        print recommend
#        print "\n"
        
#    print "End testing ..."
    
    print "Iteration:", numIteration, "mean_rank:", sumMeanRank, "max_rank:", sumMaxRank
    resStr =  str(numIteration) + " " + str(sumMeanRank) + " "+str(sumMaxRank)+"\n"
#    print "#error:", error, "percentage: ", (error+0.0)/(nUser * p)
#    resStr = "#error:"+ str(error)+", percentage: "+str( (error+0.0)/(nUser * p))
#    resStr += "\n"
    f_output = open(config.outputFile2, 'a')
    f_output.write(resStr)
    f_output.close()
