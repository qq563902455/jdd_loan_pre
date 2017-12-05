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
t_loan_sum=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_loan_sum.csv");



'''
拆分训练集，以及测试集
'''
colTrainDrop=['uid'];
for col in t_vis.columns:
    if '_11' in col:
        colTrainDrop.append(col);
dataTrainX=t_vis.drop(colTrainDrop,axis=1);
dataTrainY=t_vis['loan_amount_11'].apply(getEncryptVal);

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

#corrmat=pd.concat([dataTrainX,dataTrainY],axis=1).corr();
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

print(dataTrainX.columns)


dataTrainX['sum_amount']=dataTrainX['loan_amount_0']+dataTrainX['loan_amount_1']+dataTrainX['loan_amount_2'];
dataPre['sum_amount']=dataPre['loan_amount_0']+dataPre['loan_amount_1']+dataPre['loan_amount_2'];


'''
onehot&尺度调整 
'''

colCatList=['sex','cate_id_0','cate_id_1','cate_id_2',
#                  'param_0','param_1','param_2',
                  'pid_0','pid_1','pid_2',];
            
colScList=dataTrainX.drop(colCatList,axis=1).columns.values.tolist();       
#colCatList=[];
#colScList=[];
            
preprocessor=rawDataProcess(dataTrainX,dataPre,colCatList,colScList);

dataTrainX=preprocessor.toTrainData();
dataPre=preprocessor.toTestData();
print(dataPre.head())


'''
调参
'''
def rmseScoreCal(y, y_pred, **kwargs):
    y=pd.Series(y);
    y_pred=pd.Series(y_pred);
    return math.sqrt(mean_squared_error(y, y_pred, **kwargs));






#
##gridsearch调参
rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
lgbReg=lgb.LGBMRegressor();
paramGrid={  
        'num_leaves':[31],
        'subsample':[0.9],
        'colsample_bytree':[0.8],
        'subsample_freq':[1,2],
        'min_child_samples':[600],
        'min_child_weight':[0.001],
        'max_depth':[6],
        'random_state':[666],
        'n_estimators':[500],
        'learning_rate':[0.02],
        'subsample_for_bin':[220000],
        'min_split_gain':[0.0],
        'reg_alpha':[0],
        'reg_lambda':[0],
        'boosting_type':['gbdt'],
        'objective':['regression'],
        'verbose':[1]
};
#paramGrid={};
lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
lgbReg.fit(dataTrainX,dataTrainY);
print(lgbReg.best_score_);
print(lgbReg.best_params_);
temp=lgbReg.cv_results_;



#gridsearch调参
rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
#lgbReg=xgb.XGBRegressor();
#paramGrid={  
#        'subsample':[0.8],
#        'colsample_bylevel':[0.9],
#        'colsample_bytree':[0.9],
#        'max_depth':[5],
#        'seed':[666],
#        'learning_rate':[0.02],
#        'n_estimators':[300],
#        'min_child_weight':[5]
#};
#lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
#lgbReg.fit(dataTrainX,dataTrainY);
#print(lgbReg.best_score_);
#print(lgbReg.best_params_);
#print(lgbReg.cv_results_);
#temp=lgbReg.cv_results_

#def selfEvalMetric(y_true, y_pred):
#    return ('rmse',rmseScoreCal(y_true, y_pred),False);


#Reg=lgb.LGBMRegressor(subsample=1.0,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=600,max_depth=5,random_state=666,n_estimators=2000,learning_rate=0.02,verbose=1);
#kfold=KFold(n_splits=5,random_state=1);
#rmseScoreCalList=[];
#for kTrainIndex,kTestIndex in kfold.split(dataTrainX,dataTrainY):
#        kTrain_x=dataTrainX.iloc[kTrainIndex];  
#        kTrain_y=dataTrainY.iloc[kTrainIndex]; 
#        
#        kTest_x=dataTrainX.iloc[kTestIndex];  
#        kTest_y=dataTrainY.iloc[kTestIndex]; 
#        
#        
#        Reg.fit(kTrain_x,kTrain_y,eval_set=(kTest_x,kTest_y),eval_metric=selfEvalMetric,verbose=True,early_stopping_rounds=100);
#        #Reg.fit(kTrain_x,kTrain_y);
#        testPre=Reg.predict(kTest_x);
#        
#        rmseScore=rmseScoreCal(kTest_y,testPre);
#        print('single rmse:',rmseScore);
#        rmseScoreCalList.append(rmseScore);
#        break;
#        
#print('mean rmse:',np.array(rmseScoreCalList).mean());
     
'''
删除部分特征以达到更好的效果
'''
#estimator = lgb.LGBMRegressor(subsample=1.0,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=600,max_depth=5,random_state=666,n_estimators=450,learning_rate=0.02,verbose=1);
#selector = RFECV(estimator, step=20, cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer,verbose=True);
#selector = selector.fit(dataTrainX,dataTrainY);
#print(len(selector.grid_scores_));
#print(selector.grid_scores_);
#print(selector.get_support(indices=True))
#print(selector.n_features_)
#dataTrainX=dataTrainX.iloc[:,selector.get_support(indices=True)]
#
#
#
#modelList={
#        lgb.LGBMRegressor(subsample=1.0,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=600,max_depth=5,random_state=666,n_estimators=450,learning_rate=0.02,verbose=1),
#        xgb.XGBRegressor(subsample=0.8,colsample_bytree=0.9,colsample_bylevel=0.9,max_depth=5,seed=666,learning_rate=0.02,n_estimators=300),        
#        };
#higherModel=Ridge(random_state=888,alpha=0);
#stackerModel=stacker(modelList,higherModel,obj='reg',kFold=KFold(n_splits=5,random_state=555),kFoldHigher=KFold(n_splits=5,random_state=444));
#
#stackerModel.fit(dataTrainX,dataTrainY);
#
#for model in stackerModel.modelHigherList:
#    print(model.coef_)
##
#ans=stackerModel.predict(dataPre[dataTrainX.columns]);
#
#
#ans=pd.DataFrame({'uid':t_vis['uid'],'loan_amount':ans});
#ans.to_csv('D:/general_file_unclassified/about_code/JDD/Loan_pre/data/submit.csv',columns=['uid','loan_amount'],index=False);
#print('end');





