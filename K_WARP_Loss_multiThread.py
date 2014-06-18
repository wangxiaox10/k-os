# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Since every step is independent from each other
Use threads to accelerate the learning speed.

Author : Xiao.wang@polytechnique.edu
Date: 4 Juin, 2014
"""
# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date:   22 May, 2014


from preprocessData import * 
from model import *
from lossFunction import * 
from pickingPositiveItem import *
from predictRanking_multiThread import *
from numericalInterest import *
import numpy
import config
import thread

import threading
import time



class multiTaskLeaning_WARPLoss(threading.Thread):
    def __init__(self, nom = ''): 
        threading.Thread.__init__(self) 
        self.nom = nom 
        self.Terminated = False 
        
        
    def run(self): 
        global X
        global V
        global maxIteration
        global countIteration
        global mutex
        
        while not self.Terminated:
            lossFunc = lossFunction()
            # object numericalInterest to compute how a user likes items d
            nInterest = numericalInterest()
    
            """
            pick a positive item d using algorithm 1
            """
            (u, d) = pickingPositiveItem(X,V)
            N=0
            
            """
            pick a bar_d at random from D\Du
            """
            
            bar_Du = lossFunc.getBarDu(X, u)
            bar_d = numpy.random.choice(bar_Du)
            
            f_d_u = nInterest.f_d(d,u,X,V)
            f_bar_d_u = nInterest.f_d(bar_d, u, X, V)
            
            N += 1
            
            while( (f_bar_d_u <= f_d_u - 1 ) or (N<len(bar_Du))):
                bar_d = numpy.random.choice(bar_Du)
                f_bar_d_u = nInterest.f_d(bar_d, u, X, V)
                N += 1
                
            if (f_bar_d_u > f_d_u -1):
                """
                make a gradient step to minimize
                """
                nrank = len(bar_Du) / N
                
#                print self.nom, "learning for the ", currentIteration, " time"
                if mutex.acquire(1):
                    currentIteration = countIteration
                    alpha = 1.0 / currentIteration
                    V = lossFunc.SGD_Warp(X, V,u, d, bar_d, alpha,nrank)
                    V = lossFunc.constraintNorm(V)
                    countIteration += 1
                    
                    mutex.release()
                    
                    if( currentIteration % 20 == 0):
                        observationThread = computeLossWARP(str(currentIteration))
                        observationThread.start()
                        
            if countIteration >= maxIteration:
                self.Terminated = True
        

    def stop(self): 
        self.Terminated = True
        
class computeLossWARP(threading.Thread):
    def __init__(self, nom = ''): 
        threading.Thread.__init__(self) 
        self.nom = nom 
        
        
    def run(self): 
        print self.nom, "start observing"
        global X
        global V
        global test
        #compute aucLoss and rank predictions every few seconds. 
        lossFunc = lossFunction()
        # initial loss before the loop
        V_copy = numpy.copy(V)
#        AUCLoss = lossFunc.AUCLoss(X,V_copy)
        WARPLoss = lossFunc.combinedWARPLoss_Prediction(X,V_copy, test, self.nom)
        print self.nom, "loss:", WARPLoss
        resToWrite = str(self.nom)+ " "+ str(WARPLoss)+"\n"
        f_output = open(config.outputFile, 'a')
        f_output.write(resToWrite)
        f_output.close()
#        predictRanking(test, config.p, X, V, self.nom)
        
        
    
"""
0. constants
"""
startTime = time.time()
p = preprocessData()
inputFile = config.inputFile
(X, test) = p.getTrainingData(inputFile)
#print "X:"
#print X
print test
m = config.m
mutex=thread.allocate_lock()
countIteration = 0
maxIteration = config.maxIteration
#n number of users
n = X.shape[1]
"""
1. initialise the model
"""
# matrix of approximation to learn
V = model.model(m,n).getV()
#print "V:", V

t1 = multiTaskLeaning_WARPLoss("t1")
#t2 = multiTaskLeaning_WARPLoss("t2")
#t3 = multiTaskLeaning_WARPLoss("t3")
#t4 = multiTaskLeaning_WARPLoss("t4")
##observer = computeLossAUC("ob")
##observer.start()
t1.start()
#t2.start()
#t3.start()
#t4.start()
#
t1.join()
#t2.join()
#t3.join()
#t4.join()
#t3.start()
#t4.start()
#time.sleep(10)

lossFunc = lossFunction()
        # initial loss before the loop
WARPLoss = lossFunc.WARPLoss(X,V)


resToWrite = "#total loss:" + str(WARPLoss)+"\n"
f_output = open(config.outputFile, 'a')
f_output.write(resToWrite)
f_output.close()
#print "#finish learning", countIteration
print "#total loss:", WARPLoss
print "#time consumed:", time.time()-startTime