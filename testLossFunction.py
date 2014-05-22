# -*- coding: utf-8 -*-
from lossFunction import *
import numpy

X = array([[3,1,2,0],[4,3,4,3],[3,2,1,5], [1,6,5,2]])
V = numpy.zeros((2,4))
print X, V
f = lossFunction(X,V)
print "AUCLoss:", f.AUCLoss(0, array([0,1,2,3]))
