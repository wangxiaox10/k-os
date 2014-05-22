# -*- coding: utf-8 -*-
import numpy
import scipy.sparse
from lossFunction import *

class preprocessData:
    def __init__(self):
        # get X from raw data
        self.X = numpy.array([[3,1,2,0],[0,3,4,3],[3,0,1,5], [1,0,5,2]])
    def preprocessFile(self, inputFile):
        
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
        
        print X.shape
#        print len(nUser), len(nItem)
        self.X = X 
        return X
        
    def getTrainingData(self, inputFile):
        """
        1. Remove p = 5 known ratings
        2. return X for training
        3. compare the predicting result with extracted p ratings
        """
        
        X = self.preprocessFile(inputFile)
        p = 5
        
        nUser = X.shape[0]
        nItem = X.shape[1]
        
        lossFunc = lossFunction()
        testSet = numpy.zeros((nUser, p))
        
        for u in range(nUser):
            Du = lossFunc.getDu(X, u)
            if( len(Du) <=5 ):
                print X[u]
            testSet[u] = numpy.random.choice(Du, p)
            numpy.delete(X[u], testSet)
        self.X = X
        return (X, testSet)
            
        
    
        
    def getX(self):
        return self.X
        
#p = preprocessData()
##p.preprocessFile("/home/xiao/ProjetLibre/ml-100k/u.data")
#p.getTrainingData("/home/xiao/ProjetLibre/ml-100k/u.data")
