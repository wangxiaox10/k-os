# -*- coding: utf-8 -*-
# Author : xiao.wang@polytechnique.edu
# Date : 21 May, 2014

"""
This class defines the probability of drawing the ith position
in a list of size K.
"""

class Pdistribution:
    
    def __init__(self, i, K):
        self.i = i
        self.K = K
        print 'create Pdistribution object'
    
    def uniformeDistribution(self, C):
        return C

    """
    if j == self.i  prob(j)=1
    else            prob(j)=0
    """
    def chooseOneDistribution(self):
        return self.i
#        if j == self.i:
#            return 1
#        else:
#            return 0
    
        
