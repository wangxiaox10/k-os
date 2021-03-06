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

        self.V = numpy.random.normal(miu, sigma, (m,n))

        
    def getV(self):
        return self.V
    
        