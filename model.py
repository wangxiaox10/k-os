# -*- coding: utf-8 -*-
import numpy 
class model:
    # m is the dimension of learning matrix 
    def __init__(self, m, n):
        #initialize model parameters
        self.V = numpy.random.normal(0, 1/numpy.sqrt(m), (m,n))
    
    def getV(self):
        return self.V
        
        
        
        