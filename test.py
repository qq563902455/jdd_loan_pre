# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:28:41 2017

@author: 56390
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:59:30 2017

@author: 56390
"""

import pandas as pd;
import numpy  as np;
import seaborn as sns;
import math;
import matplotlib.pyplot as plt;

from sklearn.metrics import make_scorer;
from sklearn.metrics import mean_squared_error;
from sklearn.model_selection import GridSearchCV;
from sklearn.model_selection import KFold;
from sklearn.feature_selection import RFECV;
from sklearn.linear_model import SGDRegressor;
from sklearn.linear_model import Ridge;

from preprocess import rawDataProcess;
from stacker import stacker;
import xgboost as xgb;
import lightgbm as lgb;


def getEncryptVal(val):
    if val<0:
        return 0;
    else:
        return math.log(val+1,5);


t_vis=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_vis.csv");
t_vis=t_vis.drop(['Unnamed: 0'],axis=1);

'''
拆分训练集，以及测试集
'''
colTrainDrop=['uid'];
for col in t_vis.columns:
    if '_11' in col:
        colTrainDrop.append(col);
dataTrainX=t_vis.drop(colTrainDrop,axis=1);
dataTrainY=t_vis['t_loan_loan_amount_11'].apply(getEncryptVal);

colPreDrop=['uid'];
for col in t_vis.columns:
    if '_8' in col:
        colPreDrop.append(col);
dataPre=t_vis.drop(colPreDrop,axis=1);

'''
修改列名，方便后面的进一步处理
'''
colNameList=[];
for col in dataTrainX.columns:
    if '_8' in col:
        colNameList.append(col[0:len(col)-2]+'_0');
    elif '_9' in col:
        colNameList.append(col[0:len(col)-2]+'_1');
    elif '_10' in col:
        colNameList.append(col[0:len(col)-3]+'_2');
    else:
        colNameList.append(col);
dataTrainX.columns=colNameList;
dataPre.columns=colNameList;
colCatList=[];
#colCatList=['sex',
##                  'cate_id_0','cate_id_1','cate_id_2',
##                  'param_0','param_1','param_2',
#                  'pid_0','pid_1','pid_2',
#                  'mostly_loan_day_or_night_0','mostly_loan_day_or_night_1','mostly_loan_day_or_night_2',
#                  'mostly_loan_hour_0','mostly_loan_hour_1','mostly_loan_hour_2'];
            
for col in dataTrainX.drop(colCatList,axis=1).columns:
    if '_0' in col:
        dataTrainX[col[0:len(col)-2]+'_sum012']=dataTrainX[col[0:len(col)-2]+'_0']+dataTrainX[col[0:len(col)-2]+'_1']+dataTrainX[col[0:len(col)-2]+'_2'];
        dataPre[col[0:len(col)-2]+'_sum012']=dataPre[col[0:len(col)-2]+'_0']+dataPre[col[0:len(col)-2]+'_1']+dataPre[col[0:len(col)-2]+'_2'];
 

dataTrainX['is_there_zero_loan']=0;
dataTrainX['is_there_three_loan']=1;

dataTrainX.loc[dataTrainX['t_loan_loan_amount_sum012']==0,'is_there_zero_loan']=1;
dataTrainX.loc[dataTrainX['t_loan_loan_amount_0']==0,'is_there_three_loan']=0;
dataTrainX.loc[dataTrainX['t_loan_loan_amount_1']==0,'is_there_three_loan']=0;
dataTrainX.loc[dataTrainX['t_loan_loan_amount_2']==0,'is_there_three_loan']=0;

dataPre['is_there_zero_loan']=0;
dataPre['is_there_three_loan']=1;

dataPre.loc[dataPre['t_loan_loan_amount_sum012']==0,'is_there_zero_loan']=1;
dataPre.loc[dataPre['t_loan_loan_amount_0']==0,'is_there_three_loan']=0;
dataPre.loc[dataPre['t_loan_loan_amount_1']==0,'is_there_three_loan']=0;
dataPre.loc[dataPre['t_loan_loan_amount_2']==0,'is_there_three_loan']=0;

dataTrainX[dataTrainX['is_there_zero_loan']==1].shape[0]


def rmseScoreCal(y, y_pred, **kwargs):
    y=pd.Series(y);    
    y_pred=pd.Series(y_pred);
    return math.sqrt(mean_squared_error(y, y_pred, **kwargs));


def selfEvalMetric(y_true, y_pred):
    return ('rmse',rmseScoreCal(y_true, y_pred),False);


Reg1=lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=5,random_state=666,n_estimators=400,learning_rate=0.005,verbose=1);

Reg2=lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=9,random_state=666,n_estimators=400,learning_rate=0.02,verbose=1);

Reg3=lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=8,random_state=666,n_estimators=400,learning_rate=0.02,verbose=1);


kfold=KFold(n_splits=5,random_state=1);
rmseScoreCalList=[];

feature1List=[];
for col in dataTrainX.columns:
    if 't_loan' not in col:
        feature1List.append(col);

feature2List=[];
for col in dataTrainX.columns:
    if ('t_order' not in col) and ('t_click' not in col):
        feature2List.append(col);
                
feature3List=[];
for col in dataTrainX.columns:
        feature3List.append(col);        

for kTrainIndex,kTestIndex in kfold.split(dataTrainX,dataTrainY):
        kTrain_x=dataTrainX.iloc[kTrainIndex];  
        kTrain_y=dataTrainY.iloc[kTrainIndex]; 
        
        kTest_x=dataTrainX.iloc[kTestIndex];  
        kTest_y=dataTrainY.iloc[kTestIndex]; 
        
        
        
        trainX_1=kTrain_x[kTrain_x['is_there_zero_loan']==1][feature1List];
        trainY_1=kTrain_y[kTrain_x['is_there_zero_loan']==1];
        
        testX_1=kTest_x[kTest_x['is_there_zero_loan']==1][feature1List];
        testY_1=kTest_y[kTest_x['is_there_zero_loan']==1];
        
        
        for col in trainX_1.columns:
            if 't_loan' in col or 't_click' in col:
                trainX_1=trainX_1.drop([col],axis=1);
                testX_1=testX_1.drop([col],axis=1);
        
        
        Reg1.fit(trainX_1,trainY_1,eval_set=(testX_1,testY_1),eval_metric=selfEvalMetric,verbose=False,early_stopping_rounds=50);
#        Reg1.fit(trainX_1,trainY_1)
        testPre1=Reg1.predict(testX_1);
        
        trainX_2=kTrain_x[kTrain_x['is_there_three_loan']==1][feature2List];
        trainY_2=kTrain_y[kTrain_x['is_there_three_loan']==1];
        
        testX_2=kTest_x[kTest_x['is_there_three_loan']==1][feature2List];
        testY_2=kTest_y[kTest_x['is_there_three_loan']==1];
        
        
        Reg2.fit(trainX_2,trainY_2,eval_set=(testX_2,testY_2),eval_metric=selfEvalMetric,verbose=False,early_stopping_rounds=50);
        testPre2=Reg2.predict(testX_2);
        
        
        trainX_3=kTrain_x[kTrain_x['is_there_zero_loan']+kTrain_x['is_there_three_loan']==0][feature3List];
        trainY_3=kTrain_y[kTrain_x['is_there_zero_loan']+kTrain_x['is_there_three_loan']==0];
        
        testX_3=kTest_x[kTest_x['is_there_zero_loan']+kTest_x['is_there_three_loan']==0][feature3List];
        testY_3=kTest_y[kTest_x['is_there_zero_loan']+kTest_x['is_there_three_loan']==0];
        
        
        Reg3.fit(trainX_3,trainY_3,eval_set=(testX_3,testY_3),eval_metric=selfEvalMetric,verbose=False,early_stopping_rounds=50);
        testPre3=Reg3.predict(testX_3);
#        
#        
        pd.concat([pd.Series(testY_1),pd.Series(testY_2),pd.Series(testY_3)]).shape
        print(rmseScoreCal(testY_1,testPre1));
        print(rmseScoreCal(testY_2,testPre2));
        print(rmseScoreCal(testY_3,testPre3));
#        
        rmseScore=rmseScoreCal(pd.concat([pd.Series(testY_1),pd.Series(testY_2),pd.Series(testY_3)]),pd.concat([pd.Series(testPre1),pd.Series(testPre2),pd.Series(testPre3)]));
        print('single rmse:',rmseScore);
        rmseScoreCalList.append(rmseScore);
        
print('mean rmse:',np.array(rmseScoreCalList).mean());




#gridsearch调参
#rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
#lgbReg=lgb.LGBMRegressor();
#paramGrid={  
#        'num_leaves':[31],
#        'subsample':[0.9],
#        'colsample_bytree':[0.8],
#        'subsample_freq':[2],
#        'min_child_samples':[400],
#        'min_child_weight':[0.001],
#        'max_depth':[8,9,10],
#        'random_state':[666],
#        'n_estimators':[400],
#        'learning_rate':[0.02],
#        'subsample_for_bin':[220000],
#        'min_split_gain':[0.0],
#        'reg_alpha':[0],
#        'reg_lambda':[0],
#        'boosting_type':['gbdt'],
#        'objective':['regression'],
#        'verbose':[1]
#};
##lgbReg=SGDRegressor();
##paramGrid={'random_state':[888],
##           };
#lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
#lgbReg.fit(dataTrainX[dataTrainX['is_there_three_loan']==1],dataTrainY[dataTrainX['is_there_three_loan']==1]);
#print(lgbReg.best_score_);
#print(lgbReg.best_params_);
#temp=lgbReg.cv_results_;





