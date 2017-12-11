# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:41:27 2017

@author: 56390
"""


import pandas as pd;
import numpy  as np;
import copy as cp;
import math;
from sklearn.metrics import mean_squared_error;
from sklearn.model_selection import KFold; 

class featureGernerator:
    def __init__(self,model,dataX,dataY,kFold=KFold(n_splits=5,random_state=0),verbose=False):
        self.model=model;
        self.dataX=dataX;
        self.dataY=dataY;
        self.kFold=kFold;
        self.verbose=verbose;
    def fit(self):
        artificialFeatures=self.dataY.copy();
        for kTrainIndex,kTestIndex in self.kFold.split(self.dataX,self.dataY):
            kTrain_x=self.dataX.iloc[kTrainIndex];  
            kTrain_y=self.dataY.iloc[kTrainIndex]; 
            
            kTest_x=self.dataX.iloc[kTestIndex];  
            kTest_y=self.dataY.iloc[kTestIndex]; 
            
            for col in kTrain_y.columns:
                self.model.fit(kTrain_x,kTrain_y[col]);
                colPre=self.model.predict(kTest_x);
                if self.verbose==True:
                    print(col+' rmse: ',math.sqrt(mean_squared_error(kTest_y[col], colPre)));
                artificialFeatures[col].iloc[kTestIndex]=colPre;            
        return artificialFeatures;
            
        
