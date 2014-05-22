# -*- coding: utf-8 -*-
import numpy
import scipy.sparse

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
        
   def getTrainingData(self):
       """
       1. Remove p known ratings
       2. return X for training
       3. compare the predicting result with extracted p ratings
       """
        
    
        
    def getX(self):
        return self.X
        
p = preprocessData()
p.preprocessFile("/home/xiao/ProjetLibre/ml-100k/u.data")
