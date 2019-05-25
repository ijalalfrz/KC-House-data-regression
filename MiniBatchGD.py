# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:57:41 2019

@author: RIZAL ALFARIZI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

class MiniBatchGradientDescent:
    
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []
    batch = 0
    lr = 0
    max_iter = 0
    theta = 0
    
    def __init__(self, X_train,Y_train,X_test,Y_test, lr=0.01, max_iter=100,batch=20):
        self.X_train = X_train
        #self.X_train = np.c_[np.ones((len(X_train),1)), X_train]
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test
        self.lr = lr
        self.max_iter = max_iter
        #random theta sesuai jumlah feature + 1 untuk intercept
        self.theta = np.random.randn(self.X_train.shape[1]+1)
        self.batch = batch
        
    def costFunction(self,theta,X,y):
        "1/2m * E(h0-y)**2"
        m = len(y)
        y_pred = X.dot(theta)
        cost = (1/(2*m)) * np.sum(np.square(y_pred-y))
        
        return cost
    
    
    def estimate(self):
        m = len(self.Y_train)
        
        cost = np.zeros(self.max_iter)
        
        #theta_hist = np.zeros(max_iter)
        i = 0
        while i < self.max_iter:
            #random index data
            c = 0
            index = np.random.permutation(m)
            self.X_train = self.X_train[index]
            self.Y_train = self.Y_train[index]
            
            #kerjakan gradient descent sesuai jumlah batch
            for j in range(0,m,self.batch):
                X = self.X_train[j:j+self.batch]
                Y = self.Y_train[j:j+self.batch]
                #tambah kolom untuk tetha0
                X = np.c_[np.ones(len(X)),X]
                y_pred = np.dot(X,self.theta)
               
                error = y_pred-Y
                #update semua theta
                self.theta = self.theta - (1/m)*self.lr*(X.T.dot((error)))
                
                c += self.costFunction(self.theta,X, Y)
            
            cost[i] = c

            i+=1
            
            #print(mse_hist[i])            
        return (self.theta, cost)
    
    
    def test(self,X,Y):
        res =[]
        X = np.c_[np.ones(len(X)),X]
        i = 0
        for row in X:
            price_pred = np.dot(row,self.theta)
            res.append(price_pred)        
            i+=1
            
        data = pd.DataFrame()
        data['act'] = Y
        data['pred'] = res
        #r2 = 1 - (RSS/TSS)
        r2 = r2_score(res, Y)
        #adj-r squared untuk mengecek r2 terhadap perubahan jumlah feature
        adjr2 = 1 - (1-r2)*(len(X)-1)/(len(X)-len(X[0])-1)
        data['r2_score'] = r2
        data['adjusted_r2_score'] = adjr2
        print("R2 :", r2)
        print("adj-R2 :", adjr2)
        data.to_csv('output.csv')
            
        
        
        
        
        
        