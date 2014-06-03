# -*- coding: utf-8 -*-
# Author: xiao.wang@polytecnique.edu
# Date: 23 Mai, 2014

"""
this file implements the method to initialize the matrix V. 
More initialization methods might be further introduced for comparison.
"""

import numpy 
class model:
    # m : number of factors
    # n : number of items 
    def __init__(self, m, n):
        #initialize model parameters
        miu, sigma = 0, 1.0/numpy.sqrt(m)
#        print miu, sigma 
        
#        self.V = numpy.random.normal(m, sigma, (m,n))
        self.V = numpy.random.normal(miu, sigma, (m,n))
        #check 
#        
#        print numpy.mean(self.V), numpy.std(self.V)
#        
#        if( abs(numpy.mean(self.V) - miu) < 0.02):
#            print "mean correct"
#        else:
#            print "mean incorrect"
#        if( abs(numpy.std(self.V) - sigma) < 0.02):
#            print "deviation correct"
#        else:
#            print "deviation incorrect"
        
        
    def getV(self):
        return self.V
    
    
        



#        
#m = model(10,100)
#V = m.getV()
#print 0, 1/1.0/numpy.sqrt(10)
#print numpy.mean(V), numpy.std(V)
        