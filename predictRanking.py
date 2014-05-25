# -*- coding: utf-8 -*-
from lossFunction import * 
from pickingPositiveItem import *
from numericalInterest import *
from K_AUC_Loss import *
import numpy
import config

def predictRanking(test, p, X, V):

#    (X, V) = k_os_AUC_loss()
#    print X,V
    nUser = X.shape[0]
    
    nInterest = numericalInterest()
    lossFunc = lossFunction()
    error = 0
    for u in range(nUser):
#        print u+1
#        f_u = nInterest.f(u,X,V)
        bar_Du = lossFunc.getBarDu(X, u)
        f_u_rating_of_unknown_items = nInterest.f_set(bar_Du, u, X, V)
        f_order = numpy.argsort(f_u_rating_of_unknown_items)
        f_order = f_order[::-1] # in descending order 
        
        '''
        recommend 2*p items 
        '''
        recommend = f_order[:2*p]
        
        for i in test[u]:
            if i not in recommend:
                error += 1
        
#        print  u+1, f_order, f_u
    print "#error:", error, "percentage: ", (error+0.0)/(nUser * p)
    resStr = "#error:"+ str(error)+", percentage: "+str( (error+0.0)/(nUser * p))
    f_output = open(config.fileErrorRate, 'w')
    f_output.write(resStr)
    f_output.close()

#predictRanking()