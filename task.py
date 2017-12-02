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

from preprocess import rawDataProcess;

import lightgbm as lgb;


t_vis=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_vis.csv");
t_vis=t_vis.drop(['Unnamed: 0'],axis=1);
print(t_vis['param_8'].unique())

'''
拆分训练集，以及测试集
'''
colTrainDrop=['uid'];
for col in t_vis.columns:
    if '11' in col:
        colTrainDrop.append(col);
dataTrainX=t_vis.drop(colTrainDrop,axis=1);
dataTrainY=t_vis['loan_amount_11'];

colPreDrop=['uid'];
for col in t_vis.columns:
    if '8' in col:
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


'''
onehot&尺度调整 
'''

colCatList=['sex','cate_id_0','cate_id_1','cate_id_2',
                  'param_0','param_1','param_2',
                  'pid_0','pid_1','pid_2',];
            
colScList=dataTrainX.drop(colCatList,axis=1).columns.values.tolist();       
            
preprocessor=rawDataProcess(dataTrainX,dataPre,colCatList,colScList);

dataTrainX=preprocessor.toTrainData();
dataPre=preprocessor.toTestData();
print(dataPre.head())


'''
调参
'''

def getEncryptVal(val):
    if val<0:
        return 0;
    else:
        return math.log(val+1,5);
def rmseScoreCal(y, y_pred, **kwargs):
    y=pd.Series(y);
    y=y.apply(getEncryptVal);
    y_pred=pd.Series(y_pred);
    y_pred=y_pred.apply(getEncryptVal);
    return math.sqrt(mean_squared_error(y, y_pred, **kwargs));


#gridsearch调参
rmseScorer=make_scorer(rmseScoreCal,greater_is_better=False);
lgbReg=lgb.LGBMRegressor();
paramGrid={  
        'subsample':[0.8],
        'colsample_bytree':[0.8],
        'max_bin':[8,9,10],
        'subsample_freq':[9,10,11],
        'min_child_samples':[450,500,550],
        'max_depth':[11],
        'random_state':[666],
        'n_estimators':[800],
        'learning_rate':[0.5],
        'objective':['regression'],
        'verbose':[1]
};
lgbReg=GridSearchCV(lgbReg,paramGrid,cv=KFold(n_splits=5,random_state=1),scoring=rmseScorer);
lgbReg.fit(dataTrainX,dataTrainY);
print(lgbReg.best_score_);
print(lgbReg.best_params_);

#print(lgbReg.cv_results_)




'''
def selfEvalMetric(y_true, y_pred):
    return ('rmse',rmseScoreCal(y_true, y_pred),False);


Reg=lgb.LGBMRegressor(subsample=0.8,colsample_bytree=0.8,max_depth=10,random_state=666,boosting_type='gbdt',n_estimators=100,learning_rate=0.1,verbose=1);
kfold=KFold(n_splits=5,random_state=1);
rmseScoreCalList=[];
for kTrainIndex,kTestIndex in kfold.split(dataTrainX,dataTrainY):
        kTrain_x=dataTrainX.iloc[kTrainIndex];  
        kTrain_y=dataTrainY.iloc[kTrainIndex]; 
        
        kTest_x=dataTrainX.iloc[kTestIndex];  
        kTest_y=dataTrainY.iloc[kTestIndex]; 

        Reg.fit(kTrain_x,kTrain_y,eval_set=(kTest_x,kTest_y),eval_metric=selfEvalMetric,verbose=False);
        
        testPre=Reg.predict(kTest_x);
        rmseScore=rmseScoreCal(kTest_y,testPre);
        print('single rmse:',rmseScore);
        rmseScoreCalList.append(rmseScore);
        
        
print('mean rmse:',np.array(rmseScoreCalList).mean());
'''     



#ans=lgbReg.predict(dataPre);
#ans=pd.DataFrame({'uid':t_vis['uid'],'loan_amount':ans});
#ans['loan_amount']=ans['loan_amount'].apply(getEncryptVal);
#ans.to_csv('D:/general_file_unclassified/about_code/JDD/Loan_pre/data/submit.csv',columns=['uid','loan_amount'],index=False);
#








