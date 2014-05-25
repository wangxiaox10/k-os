# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014

from Pdistribution import * 
from preprocessData import *
from model import *
from lossFunction import *
import config

import numpy 

def pickingPositiveItem(X, V):
    """1. get obversation matrix X
    and learning model V
    """
#    nUser = 4
    nUser = X.shape[0]

    
#    print X
#    print V
    """
    2. Define a probability P 
    of drawing the ith position
    in a list of size K
    """
#    i = 1
#    K = 20
    i = config.i
    K = config.K
    
    probaDistribution = Pdistribution(i, K)
    nInterest = numericalInterest()
    
    """
    3. Pick a user u at random from the training set
    """
    
    #choose user at random
    u = numpy.random.randint(nUser)
#    print "u:", u
    
    """
    4. Pick K positive items from the Du
    """
    #pick K positive items from Du
    lossFunc = lossFunction()
    Du = lossFunc.getDu(X, u)
#    print "Du:", Du
    # OK. Got Du. Next pick K items from Du
    if( len(Du) >= K):
        items = numpy.random.choice(Du, K, replace=False)
    else:
        items = Du
#    print "items:", items
    
    """
    5.Compute f_di(u) for each selected item
    """
    f_items = nInterest.f_set(items, u, X, V)
    
    """
    6. Sort the scores by descending order.
    """
    f_order = numpy.argsort(f_items)
#    dtype = [('index', int),('f',float)]
#    index_and_f = []
#    for i in range(K):
#        index_and_f.append( (i, f_items[i]))
#    toSort_index_and_f = numpy.array(index_and_f, dtype = dtype)
#    #mapping of index and f sorted in ascending order     
#    sorted_index_and_f = numpy.sort( toSort_index_and_f, order = 'f')
#    #We need to get the sorted array in descending order 
    sorted_index_and_f = f_order[::-1]
    
#    for o_i in range(K):
#        print o_i + 1, sorted_index_and_f[o_i]
#    
#    print "i:", i, sorted_index_and_f[i-1]
    
    """
    7. Pick a position k using the distribution
    """
    res_k = probaDistribution.chooseOneDistribution()
    return (u,sorted_index_and_f[res_k-1])
    
#pickingPositiveItem()