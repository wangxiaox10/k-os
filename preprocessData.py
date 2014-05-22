# -*- coding: utf-8 -*-
import numpy

class preprocessData:
    def __init__(self):
        # get X from raw data
        self.X = numpy.array([[3,1,2,0],[0,3,4,3],[3,0,1,5], [1,0,5,2]])
    
    def getX(self):
        return self.X