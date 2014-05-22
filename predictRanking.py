# -*- coding: utf-8 -*-
from lossFunction import * 
from pickingPositiveItem import *
from numericalInterest import *
from K_AUC_Loss import *
import numpy

def predictRanking():

    (X, V) = k_os_AUC_loss()
#    print X,V
    nUser = X.shape[0]
    
    nInterest = numericalInterest()
    for u in range(nUser):
#        print u+1
        f_u = nInterest.f(u,X,V)
        f_order = numpy.argsort(f_u)
        f_order = f_order[::-1] # in descending order 
        print  u+1, f_order, f_u

predictRanking()