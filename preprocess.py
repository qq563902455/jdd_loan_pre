# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:04:08 2017

@author: 56390
"""
import pandas as pd;
import numpy  as np;
from sklearn.preprocessing import OneHotEncoder;
from sklearn.preprocessing import MinMaxScaler;


class rawDataProcess:
    def __init__(self,train_data,test_train,colCatList,colScList):
        
        assert type(train_data)==pd.DataFrame,'训练数据类型必须为DataFrame';
        assert type(test_train)==pd.DataFrame,'测试数据类型必须为DataFrame';
        assert type(colCatList)==list,'CAT类型数据所在列列表必须为list';
        assert type(colScList)==list, '需要尺度调整的数据所在列列表必须为list';
        
        
        self.train_data=train_data;
        self.test_data=test_train;
        self.enc=OneHotEncoder(dtype=np.int32);
        self.minMax=MinMaxScaler();
        self.colCatList=colCatList;
        self.colScList=colScList;
        
        if len(colCatList)>=1:
            self.enc.fit(train_data[colCatList].append(test_train[colCatList]));
        if len(colScList)>=1:
            self.minMax.fit(train_data[colScList].append(test_train[colScList]));
        
        
        
    def toTrainData(self):
        
        allList=self.colCatList+self.colScList;
        dataRe=self.train_data.drop(allList,axis=1);
        if len(self.colCatList)>=1:           
            encTransData=pd.DataFrame(self.enc.transform(self.train_data[self.colCatList]).toarray());
            for column in encTransData:
                dataRe[str(column)+'_cat']=encTransData[column];
        
        if len(self.colScList)>=1:
            scTransData=pd.DataFrame(self.minMax.transform(self.train_data[self.colScList]));
            for column in scTransData:
                dataRe[str(column)+'_sc']=scTransData[column];
            
        return dataRe;
            
    def toTestData(self):
        allList=self.colCatList+self.colScList;
        dataRe=self.test_data.drop(allList,axis=1);
        if len(self.colCatList)>=1:           
            encTransData=pd.DataFrame(self.enc.transform(self.test_data[self.colCatList]).toarray());
            for column in encTransData:
                dataRe[str(column)+'_cat']=encTransData[column];
        
        if len(self.colScList)>=1:
            scTransData=pd.DataFrame(self.minMax.transform(self.test_data[self.colScList]));
            for column in scTransData:
                dataRe[str(column)+'_sc']=scTransData[column];
            
        return dataRe;
        
             