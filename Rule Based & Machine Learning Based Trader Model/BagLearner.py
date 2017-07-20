
"""
   Bagging - a way to build an ensemble of learners.
           - train each learner on a different set of data
"""
import numpy as np
import pandas as pd
import math
import random as rand
import RTLearner as rtl
import LinRegLearner as lrl

class BagLearner:

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))   #create N(bags) number of learners
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def addEvidence(self, dataX, dataY):
        self.bagging(dataX, dataY)

    def bagging(self, dataX, dataY):
        n_prime = int(0.6 * dataX.shape[0])
        for i in xrange(0,self.bags): #iterate each bag
            #get random indices with replacement
            random_indices = np.random.choice(dataX.shape[0],size=n_prime,replace=True)
            train_X = dataX[random_indices,:] #X sample
            train_Y = dataY[random_indices]   #Y sample
            self.learners[i].addEvidence(train_X,train_Y) #invoke Random Tree for all bags

    def author(self):
        return 'ajoshi319'
    
    def query(self,points):
        Predict_y = np.zeros(shape=(points.shape[0], self.bags))
        for i in range(self.bags):
                Predict_y[:,i] = self.learners[i].query(points)
        Predict_y_mean = np.mean(Predict_y, axis=1)
        return Predict_y_mean