# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014
"""
This files implements the model to numericalise human interest
and most frequently used functions in this model. 
"""
import numpy

class numericalInterest:


#    def __init__(self):
#        self
        
    def f_d(self, d, u, X, V):
        """
        ***input***
        d: index of item( rated or unrated)
        u: index of user
        X: matrix of observation
        V: matrix of approximation
        
        ***output***
        numerical result how user u likes item i. 
        the bigger, the more u likes i. 
        """
        Du = X[u].nonzero()[0]
        res = numpy.inner(numpy.sum(V[:, Du], axis=1), V[:,d])/len(Du)
        return res
    
    def f(self, u, X, V):
        """
        output: set of interest of user u, generated by all the items. 
        """
        Du = X[u].nonzero()[0]
        res = numpy.dot(numpy.sum(V[:, Du], axis=1), V)/len(Du)
        return res
    
    def f_set(self, items, u, X, V):
        """
        output: set of interest of user u, generated by items in set items. 
        """
        Du = X[u].nonzero()[0]
        res = numpy.dot(numpy.sum(V[:, Du], axis=1), V[:, items])/len(Du)
        return res
    