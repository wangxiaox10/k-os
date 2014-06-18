# -*- coding: utf-8 -*-

# Author: xiao.wang@polytechnique.edu
# Date: 21 May, 2014

"""
This files implements most used functions. 
"""
from numericalInterest import *
#import scipy
import numpy
import config
from numpy import linalg as LA


class lossFunction:
    def __init__(self):
        self.nInterest = numericalInterest()
        
    
    def getDu(self, X, u):
        """
        return the set of positived items rated by user u
        """
        return (X[u]>0).nonzero()[0]
    
    def getBarDu(self, X, u):
        """
        return the set un unrated items for user u
        """

        return (X[u]==0).nonzero()[0]
        
        
    def SGD(self, X, V, u, d, bar_d, alpha):
        """
        One-step of stochastic gradient descend learning algorithm
        """
        
        Du = self.getDu(X,u)
        
        derivative_bar_d = numpy.sum(V[:, Du], axis=1)/len(Du)
        V[:,bar_d] = V[:,bar_d] - alpha*derivative_bar_d
        
        derivative_i = (V[:,bar_d] - V[:, d])/len(Du)
        for i in Du:
            V[:,i] = V[:, i] - alpha * derivative_i
        
#        derivative_d = (V[:,d]-V[:,bar_d] - numpy.sum(V[:, Du], axis=1))/len(Du)
        derivative_d_plus =  (- numpy.sum(V[:, Du], axis=1))/len(Du)
        V[:,d] = V[:,d] - alpha*derivative_d_plus
        
        return V
        
        
    def SGD_Warp(self, X, V, u, d, bar_d, alpha, n):
        
        Du = self.getDu(X,u)
        derivative_bar_d = numpy.sum(V[:, Du], axis=1)/len(Du)
        new_alpha = alpha * self.Phi_concave(n)
        
        V[:,bar_d] = V[:,bar_d] - new_alpha*derivative_bar_d
        
        derivative_i = (V[:,bar_d] - V[:, d])/len(Du)
        for i in Du:
            V[:,i] = V[:, i] - new_alpha*derivative_i
        
#        derivative_d = (V[:,d]-V[:,bar_d] - numpy.sum(V[:, Du], axis=1))/len(Du)
        derivative_d_plus =  (- numpy.sum(V[:, Du], axis=1))/len(Du)
        V[:,d] = V[:,d] - new_alpha * derivative_d_plus
        return V
        
    def constraintNorm_bak(self, V):
        """
        Check for each j, if the norm of Vj is smaller than C which is defined in config.py
        if not, do the normalisation: Vj = Vj*C/||Vj||
        """
        V_norm = LA.norm(V, axis=0)
        need_to_normalize = numpy.sum((V_norm> config.C).astype(float))
        
        if( need_to_normalize > 0 ):
            normalizator = max(V_norm)
#        index_to_normalize = (V_norm> 1).astype(float)
            V = V / normalizator
#        normalizator = numpy.divide(index_to_normalize, V_norm, dtype=float)
#        normalizator = normalizator * config.C
#        temp2 = 1-index_to_normalize
#        normalizator = normalizator + temp2
#        V = V * normalizator
        return V
        
    def constraintNorm(self, V):
        """
        Check for each j, if the norm of Vj is smaller than C which is defined in config.py
        if not, do the normalisation: Vj = Vj*C/||Vj||
        """
        V_norm = LA.norm(V, axis=0)
        index_to_normalize = (V_norm> config.C).astype(float)
#        index_to_normalize = (V_norm> 1).astype(float)
        normalizator = numpy.divide(index_to_normalize, V_norm, dtype=float)
        normalizator = normalizator * config.C
        temp2 = 1-index_to_normalize
        normalizator = normalizator + temp2
        V = V * normalizator
        return V
        
    def AUCLoss_u(self,u, X, V):
        """
        Computes the loss function for a single user, using AUC algorithm
        f_auc(u)
        Use numpy.meshgrid instead of two levels of loops to accelerate the compute
        """
        
#        n = V.shape[1]
#        D = numpy.arange(n)
        
#        bar_Du = list(set(D) - set(Du))
        Du = self.getDu(X, u)
        bar_Du = self.getBarDu(X, u)
        
#        fu = self.nInterest.f(u, X, V)
        nInterest = numericalInterest()
        fu = nInterest.f(u,X,V)
        fu_Du = fu[Du]
        fu_bar_Du = fu[bar_Du]
        
        res_Du, res_bar_Du = numpy.meshgrid(fu_Du, fu_bar_Du)
        funcG = 1 - res_Du + res_bar_Du
        res = sum(funcG[(funcG>0).nonzero()])
        return res
        
    def AUCLoss(self, X, V):
        """
        compute the loss of all users, using AUC. 
        """
        nUser = X.shape[0]
        loss = 0
        for u in range(nUser):
            loss += self.AUCLoss_u(u, X, V)
        return loss
        
    
    def comb_k_os_WARPLoss_Prediction(self, X, V, test, numIteration):
        nUser = X.shape[0]
        nInterest = numericalInterest()
        
        """
        1. initialisation
        """
        
        loss = 0
        sumMeanRank = 0
        sumMaxRank = 0
        precisionAt1 = 0
        precisionAt10 = 0
        
        recallAt1 = 0
        recallAt10 = 0
        
        for u in range(nUser):
            Du = self.getDu(X, u)
            bar_Du = self.getBarDu(X, u)
        
            fu = nInterest.f(u,X,V)
            
            fu_Du = fu[Du]
            fu_bar_Du = fu[bar_Du]
            
            '''
            Part 1: compute WARPLoss_u
            '''
            fu_Du_ordered = fu_Du.argsort()
            index_choisi = fu_Du_ordered[0-config.i]
            
            funcG = 1 + fu_bar_Du - fu_Du[index_choisi]
            
            res_temp = len(((funcG>=0).nonzero())[0])
            res = self.Phi_concave(res_temp)
            
            loss += res
            
            """
            Part 2: compute prediction
            """
            Du = self.getDu(X, u)
            bar_Du = self.getBarDu(X, u)
        
            fu = nInterest.f(u,X,V)
            
            fu_Du = fu[Du]
            fu_bar_Du = fu[bar_Du]
            
            
            
            f_u_testing_items = fu[test[u].astype(int)]
            fu_bar_Du.sort()
            testingItemRank = len(fu_bar_Du) - numpy.searchsorted(fu_bar_Du, f_u_testing_items)
            
#            if u == 28:
#                print "test[u]:", test[u], "f_test:", f_u_testing_items
#                f_temp = fu_bar_Du[::-1]
#                print "f_bar_Du:", f_temp[:5]
#                print "rank:", testingItemRank
            
            mean_rank = numpy.mean(testingItemRank)
            max_rank = numpy.max(testingItemRank)
            
            
            sumMeanRank += mean_rank
            sumMaxRank += max_rank
            
            
            
            precisionAt1 += numpy.count_nonzero(testingItemRank <= 1)
            precisionAt10 += numpy.count_nonzero(testingItemRank <= 10)
            
            recallAt1 += numpy.count_nonzero(testingItemRank <= 1) / (len(Du) + 0.0)
            recallAt10 += numpy.count_nonzero(testingItemRank <= 10) / (len(Du) + 0.0)
            
        averagedMeanRank = (sumMeanRank + 0.0) / nUser
        averagedMaxRank = (sumMaxRank + 0.0) / nUser
        
        precisionAt1 = (0.0 + precisionAt1) /(5 * nUser)
        precisionAt10 = (0.0 + precisionAt10) / (5 * nUser)
        
        print "Iteration:", numIteration, "mean_rank:", averagedMeanRank, "max_rank:", averagedMaxRank, "precision@1:", precisionAt1, "precision@10:", precisionAt10, "recall@1:", recallAt1, "recall@10", recallAt10
        resStr =  str(numIteration) + " " + str(averagedMeanRank) + " "+str(averagedMaxRank)+" "+str(precisionAt1)+" " + str(precisionAt10) +"\n"
    
        f_output = open(config.outputFile2, 'a')
        f_output.write(resStr)
        f_output.close()
        
        return loss
        
    def combinedWARPLoss_Prediction(self, X, V, test, numIteration):
        nUser = X.shape[0]
        nInterest = numericalInterest()
        
        """
        1. initialisation
        """
        
        loss = 0
        sumMeanRank = 0
        sumMaxRank = 0
        precisionAt1 = 0
        precisionAt10 = 0
        
        for u in range(nUser):
            Du = self.getDu(X, u)
            bar_Du = self.getBarDu(X, u)
        
            fu = nInterest.f(u,X,V)
            fu_Du = fu[Du]
            fu_bar_Du = fu[bar_Du]
            
            '''
            Part 1: compute WARPLoss_u
            '''
            
            res_Du, res_bar_Du = numpy.meshgrid(fu_bar_Du, fu_Du)
            funcG = 0 -(res_Du - 1 - res_bar_Du)
            
            #res = (funcG[(funcG>0).nonzero()])
            res_temp = numpy.bincount(((funcG>=0).nonzero())[0])
            res = self.Phi_array(res_temp)
        
            loss += sum(res)
            
            """
            Part 2: compute prediction
            """
            f_u_testing_items = nInterest.f_set(test[u].astype(int),X,V)
            fu_bar_Du.sort()
            testingItemRank = len(fu_bar_Du) - numpy.searchsorted(fu_bar_Du, f_u_testing_items)
            print "f_u_testing_items:", f_u_testing_itemsS
            print "fu_bar_Du:", fu_bar_Du[:5]
            
            mean_rank = numpy.mean(testingItemRank)
            max_rank = numpy.max(testingItemRank)
            
            
            sumMeanRank += mean_rank
            sumMaxRank += max_rank
            
            precisionAt1 += numpy.count_nonzero(testingItemRank <= 1)
            precisionAt10 += numpy.count_nonzero(testingItemRank <= 10)
                
        print "Iteration:", numIteration, "mean_rank:", sumMeanRank, "max_rank:", sumMaxRank, "precision@1:", precisionAt1, "precision@10:", precisionAt10
        resStr =  str(numIteration) + " " + str(sumMeanRank) + " "+str(sumMaxRank)+" "+str(precisionAt1)+" " + str(precisionAt10) +"\n"
    
        f_output = open(config.outputFile2, 'a')
        f_output.write(resStr)
        f_output.close()
        
        return loss
        
        
    def combinedAUCLoss_Prediction(self, X, V, test, numIteration):
        nUser = X.shape[0]
        
        nInterest = numericalInterest()
        
        """
        1. initialisation
        """
        loss = 0
        sumMeanRank = 0
        sumMaxRank = 0
        precisionAt1 = 0
        precisionAt10 = 0
        
        for u in range(nUser):
            Du = self.getDu(X, u)
            bar_Du = self.getBarDu(X, u)
        
            fu = nInterest.f(u,X,V)
            fu_Du = fu[Du]
            fu_bar_Du = fu[bar_Du]
            
            '''
            Part 1: compute AUCCLoss_u
            '''
            res_Du, res_bar_Du = numpy.meshgrid(fu_Du, fu_bar_Du)
            funcG = 1- res_Du + res_bar_Du
            res = sum(funcG[(funcG>=0).nonzero()])

            loss += res
            
            """
            Part 2: compute prediction
            """
            f_u_testing_items = fu[test[u].astype(int)]
            fu_bar_Du.sort()
            testingItemRank = len(fu_bar_Du) - numpy.searchsorted(fu_bar_Du, f_u_testing_items)
            
            mean_rank = numpy.mean(testingItemRank)
            max_rank = numpy.max(testingItemRank)
            
            
            sumMeanRank += mean_rank
            sumMaxRank += max_rank
            
            precisionAt1 += numpy.count_nonzero(testingItemRank <= 1)
            precisionAt10 += numpy.count_nonzero(testingItemRank <= 10)
                
        print "Iteration:", numIteration, "mean_rank:", sumMeanRank, "max_rank:", sumMaxRank, "precision@1:", precisionAt1, "precision@10:", precisionAt10
        resStr =  str(numIteration) + " " + str(sumMeanRank) + " "+str(sumMaxRank)+" "+str(precisionAt1)+" " + str(precisionAt10) +"\n"
    
        f_output = open(config.outputFile2, 'a')
        f_output.write(resStr)
        f_output.close()
        
        return loss
        
    
    def k_os_WARPLoss_u(self, u, X, V):
        """
        Computes the loss function for a single user, using WARP algorithm
        f_warp(u)
        """      
        nInterest = numericalInterest()
        loss = 0
        Du = self.getDu(X, u)
        bar_Du = self.getBarDu(X, u)
        
        fu = nInterest.f(u,X,V)
        fu_Du = fu[Du]
        fu_bar_Du = fu[bar_Du]
            
        '''
        Part 1: compute WARPLoss_u
        '''
        fu_Du_ordered = fu_Du.argsort()
        index_choisi = fu_Du_ordered[0-config.i]
            
        funcG = 1 + fu_bar_Du - fu_Du[index_choisi]
            
        res_temp = len(((funcG>=0).nonzero())[0])
        res = self.Phi_concave(res_temp)
            
        loss += res
        
        return loss
        
    def WARPLoss_u(self, u, X, V):
        """
        Computes the loss function for a single user, using WARP algorithm
        f_warp(u)
        """        
        nInterest = numericalInterest()
        
        loss = 0
        
        Du = self.getDu(X, u)
        bar_Du = self.getBarDu(X, u)
        
        fu = nInterest.f(u,X,V)
        fu_Du = fu[Du]
        fu_bar_Du = fu[bar_Du]
            
        '''
        Part 1: compute WARPLoss_u
        '''
            
        res_Du, res_bar_Du = numpy.meshgrid(fu_bar_Du, fu_Du)
#        funcG = res_Du -1- res_bar_Du
        funcG = 0 -(res_Du - 1 - res_bar_Du)
            
            #res = (funcG[(funcG>0).nonzero()])
        res_temp = numpy.bincount(((funcG>=0).nonzero())[0])
        res = self.Phi_array(res_temp)

        loss += sum(res)
            
#        resLoss = 0
#        for d in self.getDu(X, u):
#            resLoss += self.Phi_concave(self.rank(X,V, d, u))

        return loss

    def k_os_WARPLoss(self, X, V):
        """
        compute the loss of all users, using AUC. 
        """
        nUser = X.shape[0]
        loss = 0
        for u in range(nUser):
            loss += self.k_os_WARPLoss_u(u, X, V)
            
        return loss

        
        
    def WARPLoss(self, X, V):
        """
        compute the loss of all users, using AUC. 
        """
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
        
    def Phi_array(self, a):
        res = numpy.zeros(len(a))
        
        for i in range( len(a)):
            res[i] = self.Phi_concave(a[i])
        return res
        
    def Phi_concave(self, eta):
        """
        the function converts the rank of a positive item d to a weight
        Phi(eta) = sum_{i=1}^{eta}(1/i)
        """

        res = 0.0
        for i in range(1,eta+1):
            res += 1.0/(i)
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
                

####Test####
#data = numpy.genfromtxt(config.inputFile, names=['user','item','rating'])
#Users = data['user']
#Items = data['item']
#Rating = data['rating']
#        
#nUser = numpy.unique(Users)
#nItem = numpy.unique(Items)
#        
#X = scipy.sparse.coo_matrix((Rating, (Users,Items)),shape=(max(Users)+1, max(Items)+1))
#X = X.toarray()
#
#sumRatingPerUser = X.sum(axis=1)
#removeUser = (sumRatingPerUser==0).nonzero()[0]
#X = numpy.delete(X, removeUser, axis=0)
#        
#sumRatingPerItem = X.sum(axis = 0)
#removeItem = (sumRatingPerItem==0).nonzero()[0]
#X = numpy.delete(X, removeItem, axis=1)
#        
#print X
#
#lf = lossFunction()
#
#V = lf.SGD(X,V,0,0,1,0.1)
#lf.AUCLoss_u(0,X,V)

#X = array([[ 1.,  0.,  1.,  1.,  1.],
#       [ 1.,  1.,  1.,  0.,  1.],
#       [ 0.,  0.,  1.,  1.,  1.],
#       [ 1.,  1.,  1.,  0.,  0.],
#       [ 0.,  1.,  1.,  1.,  1.]])
#
#V = array([[  0.9949356 ,  -1.04369135,   0.66468231,   0.6443851 ,   0.9506672 ],
#       [  0.95058917,  -1.57604338,   1.25349777,   1.31795382,
#          1.65435876],
#       [  4.89761307,  -6.82770325,   6.52956682,   5.3058573 ,
#          6.33985457],
#       [-11.89334796,  15.5720221 , -13.49552482, -12.56400037, -14.1265795 ],
#       [ -9.98136685,  13.57053829, -11.81116537, -11.36548843,
#        -11.79430569]])
#        
#
#
#lf = lossFunction()
#Du = lf.getDu(X, u)
#bar_Du = lf.getBarDu(X, u)
#        
#nInterest = numericalInterest()
#fu = nInterest.f(u,X,V)
#fu_Du = fu[Du]
#fu_bar_Du = fu[bar_Du]
#        
#res_Du, res_bar_Du = numpy.meshgrid(fu_Du, fu_bar_Du)
#funcG = 1 - res_Du + res_bar_Du
#res = sum(funcG[(funcG>0).nonzero()])
#print res