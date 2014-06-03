# -*- coding: utf-8 -*-
# Author: xiao.wang@polytechnique.edu
# Date: 24 Mai, 2014

"""
The file preprocesses data from file, 
and generate learning matrix X and test matrix test for further use.
"""

import numpy
import scipy.sparse
from lossFunction import *
import config

class preprocessData:
    def preprocessFile(self, inputFile):
        
        """
        In the file, each line is stored in the form: user item rating
        """
        data = numpy.genfromtxt(inputFile, names=['user','item','rating'])
        Users = data['user']
        Items = data['item']
        Rating = data['rating']
        
        nUser = numpy.unique(Users)
        nItem = numpy.unique(Items)
        
        X = scipy.sparse.coo_matrix((Rating, (Users,Items)),shape=(max(Users)+1, max(Items)+1))
        X = X.toarray()
        
        sumRatingPerUser = X.sum(axis=1)
        removeUser = (sumRatingPerUser==0).nonzero()[0]
        X = numpy.delete(X, removeUser, axis=0)
        
        sumRatingPerItem = X.sum(axis = 0)
        removeItem = (sumRatingPerItem==0).nonzero()[0]
        X = numpy.delete(X, removeItem, axis=1)
        
        self.X = X 
        return X
        
    def getTrainingData(self, inputFile):
        """
        1. For each user u, remove p = 5 known ratings from X[u]
        2. return X for training, testSet for testing
        3. In test period(in predictRanking.py) we need to compare the predicting result with extracted p ratings
        """
        
        X = self.preprocessFile(inputFile)
        p = config.p
        
        nUser = X.shape[0]
        nItem = X.shape[1]
        
        lossFunc = lossFunction()
        testSet = numpy.zeros((nUser, p))
        
        for u in range(nUser):
            Du = lossFunc.getDu(X, u)
            if( len(Du) <=p ):
                print X[u]
            testSet[u] = numpy.random.choice(Du, p)
            
            for test_u in testSet[u]:
                X[u, test_u] = 0

            
        self.X = X
        return (X, testSet)
        
    def getX(self):
        return self.X
        
