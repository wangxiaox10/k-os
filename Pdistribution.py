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

    
    def uniformeDistribution(self, C):
        return C

    
    def chooseOneDistribution(self):
        return self.i
    
        
