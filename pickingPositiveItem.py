# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014

"""
This file implements the Algorithm1 in the paper. 
"""

from Pdistribution import * 
from preprocessData import *
from model import *
from lossFunction import *
from numericalInterest import * 
import config

import numpy 

def pickingPositiveItem(X, V):
    """1. get obversation matrix X
    and learning model V
    """
    # number of users 
    nUser = X.shape[0]


    """
    2. Define a probability P 
    of drawing the ith position
    in a list of size K
    """
    i = config.i
    K = config.K
    
    probaDistribution = Pdistribution(i, K)
    nInterest = numericalInterest()
    
    """
    3. Pick a user u at random from the training set
    """
    
    u = numpy.random.randint(nUser)
    
    """
    4. Pick K positive items from the Du
    """
    lossFunc = lossFunction()
    Du = lossFunc.getDu(X, u)

    # OK. Got Du. Next pick K items from Du
    if( len(Du) >= K):
        items = numpy.random.choice(Du, K, replace=False)
    else:
        items = Du

    
    """
    5.Compute f_di(u) for each selected item
    """
    f_items = nInterest.f_set(items, u, X, V)
    
    """
    6. Sort the scores by descending order.
    """
    f_order = numpy.argsort(f_items)
    sorted_index_and_f = f_order[::-1]
    
    
    """
    7. Pick a position k using the distribution
    """
    res_k = probaDistribution.chooseOneDistribution()
    return (u,items[sorted_index_and_f[res_k-1]])
    