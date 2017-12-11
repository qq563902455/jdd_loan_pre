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

t_label_list=['t_loan_loan_amount_8','t_loan_loan_amount_9','t_loan_loan_amount_10','t_loan_loan_amount_11'];

'''
t_loan部分的一些处理
'''
#t_loan_list=[];
#for col in t_vis.columns:
#    if 't_loan' in col:
#        t_loan_list.append(col);
#t_loan_corr_table=t_vis[t_loan_list].corr();
#
#corr_rank=pd.Series();
#temp=t_loan_corr_table[t_label_list].abs().sum(axis=1);
#for col in t_loan_corr_table.columns:
#    if '_8' in col:
#        corr_rank[col[0:(len(col)-2)]]=temp[col[0:(len(col)-2)]+'_8']+temp[col[0:(len(col)-2)]+'_9']+temp[col[0:(len(col)-2)]+'_10']+temp[col[0:(len(col)-2)]+'_11'];
#
#t_loan_remain=corr_rank.sort_values(ascending=False).head(40).index.tolist();
#print('t_loan_remian: ',len(t_loan_remain));
#t_loan_remainlist=[];
#for col in t_loan_remain:
#    t_loan_remainlist.append(col+'_8');
#    t_loan_remainlist.append(col+'_9');
#    t_loan_remainlist.append(col+'_10');
#    t_loan_remainlist.append(col+'_11');
#
#dropList=[];    
#for col in  t_loan_list:
#    if col not in t_loan_remainlist:
#        dropList.append(col);
#t_vis=t_vis.drop(dropList,axis=1);


'''
t_click部分的处理
'''


t_click_list=[];
for col in t_vis.columns:
    if 't_click' in col:
        t_click_list.append(col);
corr_table=t_vis[t_click_list+t_label_list].corr();

corr_rank=pd.Series();
temp=corr_table[t_label_list].abs().sum(axis=1);
for col in corr_table.columns:
    if '_8' in col:
        corr_rank[col[0:(len(col)-2)]]=temp[col[0:(len(col)-2)]+'_8']+temp[col[0:(len(col)-2)]+'_9']+temp[col[0:(len(col)-2)]+'_10']+temp[col[0:(len(col)-2)]+'_11'];

t_click_remain=corr_rank.sort_values(ascending=False).head(30).index.tolist();
print('t_click_remian: ',len(t_click_remain));

t_click_remainlist=[];
for col in t_click_remain:
    t_click_remainlist.append(col+'_8');
    t_click_remainlist.append(col+'_9');
    t_click_remainlist.append(col+'_10');
    t_click_remainlist.append(col+'_11');

dropList=[];    
for col in  t_click_list:
    if col not in t_click_remainlist:
        dropList.append(col);
t_vis=t_vis.drop(dropList,axis=1);



'''
t_order部分的处理
'''    

t_order_list=[];
for col in t_vis.columns:
    if 't_order' in col:
        t_order_list.append(col);
corr_table=t_vis[t_order_list+t_label_list].corr();

corr_rank=pd.Series();
temp=corr_table[t_label_list].abs().sum(axis=1);
for col in corr_table.columns:
    if '_8' in col:
        corr_rank[col[0:(len(col)-2)]]=temp[col[0:(len(col)-2)]+'_8']+temp[col[0:(len(col)-2)]+'_9']+temp[col[0:(len(col)-2)]+'_10']+temp[col[0:(len(col)-2)]+'_11'];

t_order_remain=corr_rank.sort_values(ascending=False).head(1).index.tolist();
print('t_click_remian: ',len(t_order_remain));

t_order_remainlist=[];
for col in t_order_remain:
    t_order_remainlist.append(col+'_8');
    t_order_remainlist.append(col+'_9');
    t_order_remainlist.append(col+'_10');
    t_order_remainlist.append(col+'_11');

dropList=[];    
for col in  t_order_list:
    if col not in t_order_remainlist:
        dropList.append(col);
t_vis=t_vis.drop(dropList,axis=1);



'''
拆分训练集，以及测试集
'''
colTrainDrop=['uid'];
for col in t_vis.columns:
    if '_11' in col:
        colTrainDrop.append(col);
dataTrainX=t_vis.drop(colTrainDrop,axis=1);
dataTrainY=t_vis[t_label_list[3]].apply(getEncryptVal);

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

possibleColCatList=['sex','t_loan_mostly_loan_day_or_night_0','t_loan_mostly_loan_day_or_night_1','t_loan_mostly_loan_day_or_night_2',
                 't_loan_mostly_loan_hour_0','t_loan_mostly_loan_hour_1','t_loan_mostly_loan_hour_2'];

colCatList=[];
for col in possibleColCatList:
    if col in dataTrainX.columns:
        colCatList.append(col);

#colCatList=[];
            
for col in dataTrainX.drop(colCatList,axis=1).columns:
    if '_0' in col:
        dataTrainX[col[0:len(col)-2]+'_sum012']=dataTrainX[col[0:len(col)-2]+'_0']+dataTrainX[col[0:len(col)-2]+'_1']+dataTrainX[col[0:len(col)-2]+'_2'];
        dataPre[col[0:len(col)-2]+'_sum012']=dataPre[col[0:len(col)-2]+'_0']+dataPre[col[0:len(col)-2]+'_1']+dataPre[col[0:len(col)-2]+'_2'];
            
#colScList=dataTrainX.drop(colCatList,axis=1).columns.values.tolist();       
#            
#preprocessor=rawDataProcess(dataTrainX,dataPre,colCatList,colScList);
#
#dataTrainX=preprocessor.toTrainData();
#dataPre=preprocessor.toTestData();
#print(dataPre.shape);


'''
调参
'''
def rmseScoreCal(y, y_pred, **kwargs):
    y=pd.Series(y);    
    y_pred=pd.Series(y_pred);
    return math.sqrt(mean_squared_error(y, y_pred, **kwargs));






#
#gridsearch调参
rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
lgbReg=lgb.LGBMRegressor();
paramGrid={  
        'num_leaves':[31],
        'subsample':[0.9],
        'colsample_bytree':[0.8],
        'subsample_freq':[2],
        'min_child_samples':[400],
        'min_child_weight':[0.001],
        'max_depth':[7],
        'random_state':[666],
        'n_estimators':[80],
        'learning_rate':[0.1],
        'subsample_for_bin':[220000],
        'min_split_gain':[0.0],
        'reg_alpha':[1.0],
        'reg_lambda':[0.7],
        'boosting_type':['gbdt'],
        'objective':['regression'],
        'verbose':[1]
};
#lgbReg=SGDRegressor();
#paramGrid={'random_state':[888],
#           };
lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
lgbReg.fit(dataTrainX,dataTrainY);
print(lgbReg.best_score_);
print(lgbReg.best_params_);
temp=lgbReg.cv_results_;



#gridsearch调参
#rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
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
#
#lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
#lgbReg.fit(dataTrainX,dataTrainY);
#print(lgbReg.best_score_);
#print(lgbReg.best_params_);
#print(lgbReg.cv_results_);
#temp=lgbReg.cv_results_

def selfEvalMetric(y_true, y_pred):
    return ('rmse',rmseScoreCal(y_true, y_pred),False);


Reg=lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=7,random_state=666,n_estimators=2000,reg_alpha=1.0,reg_lambda=0.7,learning_rate=0.02,verbose=1);
kfold=KFold(n_splits=5,random_state=1);
rmseScoreCalList=[];
for kTrainIndex,kTestIndex in kfold.split(dataTrainX,dataTrainY):
        kTrain_x=dataTrainX.iloc[kTrainIndex];  
        kTrain_y=dataTrainY.iloc[kTrainIndex]; 
        
        kTest_x=dataTrainX.iloc[kTestIndex];  
        kTest_y=dataTrainY.iloc[kTestIndex]; 
        
        
        Reg.fit(kTrain_x,kTrain_y,eval_set=(kTest_x,kTest_y),eval_metric=selfEvalMetric,verbose=True,early_stopping_rounds=50);
#        print('best iteration: ',Reg.best_iteration_);
#        Reg.fit(kTrain_x,kTrain_y);
        testPre=Reg.predict(kTest_x);
        
        rmseScore=rmseScoreCal(kTest_y,testPre);
        print('single rmse:',rmseScore);
        rmseScoreCalList.append(rmseScore);
        
print('mean rmse:',np.array(rmseScoreCalList).mean());
     
temp=pd.Series(Reg.feature_importances_);
temp.index=dataTrainX.columns;
'''
删除部分特征以达到更好的效果
'''
estimator = lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=7,random_state=666,n_estimators=275,reg_alpha=1.0,reg_lambda=0.7,learning_rate=0.02,verbose=1);
selector = RFECV(estimator, step=20, cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer,verbose=True);
selector = selector.fit(dataTrainX,dataTrainY);
print(len(selector.grid_scores_));
print(selector.grid_scores_);
print(selector.get_support(indices=True))
print(selector.n_features_)
#
#
#temp=selector.grid_scores_;
#for i in temp:
#    print(i);
##
##
##
#dataTrainX=dataTrainX.iloc[:,selector.get_support(indices=True)]
#
#
#modelList=[
#        lgb.LGBMRegressor(subsample=0.9,colsample_bytree=0.8,subsample_for_bin=220000,min_child_samples=400,subsample_freq=2,max_depth=8,random_state=666,n_estimators=350,learning_rate=0.02,verbose=1),
#        xgb.XGBRegressor(subsample=0.8,colsample_bytree=0.9,colsample_bylevel=0.9,max_depth=5,seed=666,learning_rate=0.02,n_estimators=300),        
#        ];
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
#ans.loc[ans['loan_amount']<0,['loan_amount']]=0;
#ans.to_csv('D:/general_file_unclassified/about_code/JDD/Loan_pre/data/submit.csv',columns=['uid','loan_amount'],index=False);
#print('end');


