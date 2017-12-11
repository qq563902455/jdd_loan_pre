# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:47:17 2017

@author: 56390
"""
import pandas as pd;
import numpy  as np;
import seaborn as sns;
import matplotlib.pyplot as plt;

import math;


t_loan=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_loan.csv");


t_loan['loan_amount']=5**t_loan['loan_amount']-1;
t_loan.index=pd.to_datetime(t_loan['loan_time']);
t_loan['dayofweek']=t_loan.index.dayofweek;
t_loan['day']=t_loan.index.day;
t_loan['hour']=t_loan['loan_time'].apply(lambda x:x[11:13]).astype(int);

t_loan['hour_cat']=0;
t_loan.loc[t_loan['hour']>=7,'hour_cat']+=1;
t_loan.loc[t_loan['hour']>=16,'hour_cat']+=1;


t_loan['day_cat']=0;
t_loan.loc[t_loan['day']>10,'day_cat']+=1;
t_loan.loc[t_loan['day']>20,'day_cat']+=1;


t_loan['dayofweek_cat']=0;
t_loan.loc[t_loan['dayofweek']==0,'dayofweek_cat']=1;
t_loan.loc[t_loan['dayofweek']==6,'dayofweek_cat']=1;



num1=len(t_loan['hour_cat'].unique());
num2=len(t_loan['day_cat'].unique());
num3=len(t_loan['dayofweek_cat'].unique());

t_loan['hour_day_week_cat']=t_loan['hour_cat']*num3*num2+t_loan['day_cat']*num3+t_loan['dayofweek_cat']

t_loan['month']=t_loan['loan_time'].apply(lambda x:x[5:7]).astype(int);

tempGroupby=t_loan.groupby(by=['uid','month'],as_index=False);
t_user_loan=tempGroupby['loan_amount'].sum();
t_user_loan['plannum_amount']=tempGroupby['plannum'].sum()['plannum'];
t_user_loan['loan_average']=tempGroupby['loan_amount'].mean()['loan_amount'];
t_user_loan['plannum_average']=tempGroupby['plannum'].mean()['plannum'];
t_user_loan['burden_per_m']=t_user_loan['loan_amount']/t_user_loan['plannum_amount'];
t_user_loan['loan_num']=tempGroupby['loan_time'].count()['loan_time'];

for cat in ['hour_cat','dayofweek_cat','day_cat','hour_day_week_cat']:
      tempGroupby=t_loan.groupby(by=['uid','month',cat],as_index=False);
      t_temp=tempGroupby['loan_amount'].sum()
      t_temp['plannum_amount']=tempGroupby['plannum'].sum()['plannum'];
      t_temp['loan_average']=tempGroupby['loan_amount'].mean()['loan_amount'];
      t_temp['plannum_average']=tempGroupby['plannum'].mean()['plannum'];
      t_temp['burden_per_m']=t_user_loan['loan_amount']/t_user_loan['plannum_amount'];
      t_temp['loan_num']=tempGroupby['loan_time'].count()['loan_time'];
      rawNameList=t_temp.columns;
      for i in t_temp[cat].unique():
            t_temp.columns=rawNameList;
            for col in t_temp:
                  if col not in ['uid','month',cat]:
                        t_temp=t_temp.rename(columns={col:cat+str(i)+'_'+col});
            t_user_loan=pd.merge(t_user_loan,t_temp[t_temp[cat]==i].drop([cat],axis=1),how='left',on=['uid','month']);
      t_user_loan=t_user_loan.fillna(0);


for cat in ['hour_cat','dayofweek_cat','day_cat','hour_day_week_cat','hour_day_week_cat']:
      ratioColList=[];
      for i in t_loan[cat].unique():
            t_user_loan[cat+str(i)+'_loan_ratio']=t_user_loan[cat+str(i)+'_loan_amount']/t_user_loan['loan_amount'];
            t_user_loan['is_'+cat+str(i)+'_loan_max']=0;
            ratioColList.append(cat+str(i)+'_loan_ratio');
      t_user_loan[cat+'_loan_ratio_max']=t_user_loan[ratioColList].max(axis=1);
      
      for i in t_loan[cat].unique():
            t_user_loan.loc[t_user_loan[cat+'_loan_ratio_max']==t_user_loan[cat+str(i)+'_loan_ratio'],'is_'+cat+str(i)+'_loan_max']=1;
      
      
      
      
      




















