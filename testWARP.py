# -*- coding: utf-8 -*-
# Author : xiao.wang@polytechnique.edu
# Date : 26 May, 2014

"""
This file implements Algorithm2 mentioned in the paper. 
To be tested. 
"""

from preprocessData import *
#from model import *
from K_WARP_Loss import *
from predictRanking import * 
import config

def testWARP():
    p = preprocessData()
#    inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
    inputFile = config.inputFile
    (X, test) = p.getTrainingData(inputFile)
#    m = 20
    m = config.m
    
    V = k_os_AUC_loss(X, m)
    predictRanking(test, config.p, X, V)
    
testWARP()
