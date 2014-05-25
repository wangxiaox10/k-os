# -*- coding: utf-8 -*-
from preprocessData import *
#from model import *
from K_AUC_Loss import *
from predictRanking import * 
import config

def testAUC():
    p = preprocessData()
#    inputFile = "/home/xiao/ProjetLibre/ml-100/u.data"
    inputFile = config.inputFile
    (X, test) = p.getTrainingData(inputFile)
#    m = 20
    m = config.m
    
    V = k_os_AUC_loss(X, m)
    predictRanking(test, config.p, X, V)
    
testAUC()