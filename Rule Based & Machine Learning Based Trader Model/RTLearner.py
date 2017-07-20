import numpy as np
import pandas as pd
import math
import random as rand

class RTLearner():

    def __init__(self, leaf_size = 1, verbose=False): # constructor
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return 'ajoshi319'

    def addEvidence(self, trainX, trainY): # training step
        self.decision_tree(trainX, trainY)

    def get_RandomSplit_Values(self,data):  #keep timing performance but slice performace degrade
        
        feature_index =  rand.randrange(0, stop=data.shape[1] - 1)
        Nfeature = data[:,feature_index]
        Rand_Twofeatures = rand.sample(Nfeature,2) #Randomly selecting two samples of data
        split_val = np.mean(Rand_Twofeatures) #taking the mean of their Xi values.

        return feature_index, split_val

    def decision_tree(self, trainX, trainY ):
        leaf = -1
        Nan = 0

        if trainX.shape[0]<= self.leaf_size:
            self.tree = np.array([leaf, np.mean(trainY), Nan, Nan]) #return [leaf, np.mean(y_train), NA, NA]    #take a mean of y
            return self.tree

        unique_values = np.unique(trainY)  #if all data.y same
        if len(unique_values) == 1:
            self.tree = np.array([leaf, np.mean(trainY), Nan, Nan])   #return[NA, data.y, NA, NA]- return that label, left/right trees are NA
            return self.tree 

        feature_index, split_val = self.get_RandomSplit_Values(trainX)

        while split_val >= np.amax(trainX[:,feature_index]): #make sure spilt_val less than max of datapoint
            feature_index, split_val = self.get_RandomSplit_Values(trainX) 

        Left_Direction=trainX[:,feature_index]<=split_val  # Left

        left_trainX = trainX[Left_Direction]
        left_trainY = trainY[Left_Direction]

        right_trainX = trainX[~Left_Direction]  #inverse of left= right
        right_trainY = trainY[~Left_Direction] #inverse of left= right

        lefttree = self.decision_tree(left_trainX, left_trainY)
        righttree = self.decision_tree(right_trainX, right_trainY)

        Length_leftTree = len(lefttree.shape) 
        if Length_leftTree == 1: 
            lefttree_size_offset = 2
        else:
            lefttree_size_offset = lefttree.shape[0] + 1

        root = [feature_index, split_val, 1, lefttree_size_offset]
        self.tree = np.vstack((root, lefttree, righttree))

        return  self.tree

    def query(self, data):
        leaf = -1
        predict_y = np.zeros(data.shape[0],)  #create an zeros array
        SpiltVal_col = self.tree[:,[1]] #  Spilt_val column
        LeafVal_col = self.tree[:,0] # First column or leaf column
        for i in range(0, data.shape[0]): #iterate each row 
            k =0 #row pointer 
            while (LeafVal_col[k]) != leaf:  #check if first col is not leaf
                if (data[ i, LeafVal_col[k] ]) <= ( SpiltVal_col[k]): # check if X val is less than spilt_val
                    k = k + int(self.tree[k,2])
                else:
                    k = k + int(self.tree[k,3])
                if k >= len(self.tree):
                    k =-1
            predict_y[i] = (self.tree[k,1])  #results
        return predict_y
