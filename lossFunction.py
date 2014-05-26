# -*- coding: utf-8 -*-

# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014
from numericalInterest import *
import numpy
import config
from numpy import linalg as LA

class lossFunction:
    def __init__(self):
#        self.X = X 
#        self.V = V
        "number of items"
#        self.n = V.shape[1]
        self.nInterest = numericalInterest()
        
    """
    return the set of positived items rated by user u
    """
    def getDu(self, X, u):
        return (X[u]>0).nonzero()[0]
    
    """
    return the set un unrated items for user u
    """
    def getBarDu(self, X, u):
        return (X[u]==0).nonzero()[0]
        
    
    """
    a step of stochastic gradient descend learning algorithm
    """
    def SGD(self, X, V, u, d, bar_d, alpha):
        Du = self.getDu(X,u)
        derivative_bar_d = numpy.sum(V[:, Du], axis=1)/len(Du)
        derivative_d = (V[:,d]-V[:,bar_d] - numpy.sum(V[:, Du], axis=1))/len(Du)
        
        V[:,d] = V[:,d] - alpha*derivative_d
        V[:,bar_d] = V[:,bar_d] - alpha*derivative_bar_d
        return V
        
    def constraintNorm(self, V, C):
        V_norm = LA.norm(V, axis=0)
        index_to_normalize = (V_norm>C).astype(float)
        normalizator = numpy.divide(index_to_normalize, V_norm, dtype=float)
        normalizator = normalizator * C
        temp2 = 1-index_to_normalize
        normalizator = normalizator + temp2
#        print "nomalizator:", normalizator
        V = V * normalizator
        return V
        
    def AUCLoss_u(self,u, X, V):
        Du = self.getDu(X, u)
        
        n = V.shape[1]
        D = numpy.arange(n)
        
        bar_Du = list(set(D) - set(Du))
        
        fu = self.nInterest.f(u, X, V)
#        print "fu:", fu, "Du:", Du
        fu_Du = fu[Du]
        fu_bar_Du = fu[bar_Du]
        
        res_Du, res_bar_Du = numpy.meshgrid(fu_Du, fu_bar_Du)
        funcG = 1 - res_Du + res_bar_Du
        res = sum(funcG[(funcG>0).nonzero()])
        return res
        
    def AUCLoss(self, X, V):
        nUser = X.shape[0]
        loss = 0
        for u in range(nUser):
            loss += self.AUCLoss_u(u, X, V)
        return loss
            
    def WARPLoss_u(self, u, X, V):
        resLoss = 0
        for d in self.getDu(X, u):
            resLoss += self.Phi_concave( self.rank(X,V, d, u))
            
        return resLoss
        
    def WARPLoss(self, X, V):
        nUser = X.shape[0]
        loss = 0
        for u in range(nUser):
            loss += self.WARPLoss_u(u, X, V)
            
        return loss
    
    def Phi_constant(self, eta, C):
        """
        the function converts the rank of a positive item d to a weight
        Phi(eta) = C * eta
        """
        return eta * C
        
    def Phi_concave(self, eta):
        """
        the function converts the rank of a positive item d to a weight
        Phi(eta) = sum_{i=1}^{eta}(1/i)
        """

        res = 0.0
        for i in range(eta):
            res += 1.0/(i+1)
        return res
            
    def rank(self, X, V, d, u ):
        nInterest = numericalInterest()
        f_d_u = nInterest.f_d(d, u, X, V)
        
        D_bar_u = self.getBarDu(X, u)
        
        resCount = 0
        f_bar_d_u = nInterest.f_set(D_bar_u, u, X, V)
        
        distance = f_d_u-1 - f_bar_d_u
        """
        Indicator function: use len instead of numpy.sum
        """
        resCount = len((distance>=0).nonzero()[0])
        
        return resCount
                
            
        
    #def WARPLoss():
        
#X = numpy.array([[3,1,2,0],[4,3,4,3],[3,2,1,5], [1,6,5,2]])
#V = numpy.zeros((2,4))
#print X, V
#f = lossFunction()
#print "AUCLoss:", f.AUCLoss(0, X, V)
