# -*- coding: utf-8 -*-
from preprocessData import *
#from model import *
from K_AUC_Loss import *
from predictRanking import * 
def testAUC():
    p = preprocessData()
    inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
    (X, test) = p.getTrainingData(inputFile)
    m = 20
    V = k_os_AUC_loss(X, m)
    predictRanking(test, 5, X, V)
    
testAUC()