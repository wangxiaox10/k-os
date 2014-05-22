# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014
import numpy

class numericalInterest:

#this class stores the matrix V
#It's used after preprocessing.
    def __init__(self):
        " V is a m*|D| matrix"  
#        self.X = X 
#        self.V = V
        print "create numericalInterst object"
        
    def f_d(self, d, u, X, V):
        Du = X[u].nonzero()[0]
        res = numpy.inner(numpy.sum(V[:, Du], axis=1), V[:,d])/len(Du)
        return res
    
    def f(self, u, X, V):
        Du = X[u].nonzero()[0]
        res = numpy.dot(numpy.sum(V[:, Du], axis=1), V)/len(Du)
        return res
    
    def f_set(self, items, u, X, V):
        Du = X[u].nonzero()[0]
        res = numpy.dot(numpy.sum(V[:, Du], axis=1), V[:, items])/len(Du)
        return res
    

#test 
#X = array([[3,1,2,3],[4,3,4,3],[3,2,1,5], [1,6,5,2]])
#V = numpy.zeros((2,4))
#nInterest = numericalInterest()
#print nInterest.f(0, X, V)
