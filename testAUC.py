# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 24 Mai, 2014

"""
This files tests if the K_AUC_Loss function works correctly.
testAUC tests the whole processus. More tests are to be written 
to test single functions or small group of functions
to ensure the correctness.
"""

from preprocessData import *
from K_AUC_Loss import *
from predictRanking import * 
import config
import showOutput

def testAUC():
    p = preprocessData()
    inputFile = config.inputFile
    (X, test) = p.getTrainingData(inputFile)
    print "X:"
    print X
    print test
    m = config.m
    
    
    V = k_os_AUC_loss(X, m, test)
    
    predictRanking(test, config.p, X, V)
    
testAUC()
showOutput.showOutput()
