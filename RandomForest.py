#!/usr/bin/env python
# coding: utf-8

#################################random forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def RF(train, dataset_num):
    #gender
    onehot = pd.get_dummies(train['gender'],prefix='gender')
    train = train.drop('gender',1)
    train = pd.concat([train,onehot],axis=1)
    #marriage
    onehot = pd.get_dummies(train['marriage'],prefix='marriage')
    train = train.drop('marriage',1)
    train = pd.concat([train,onehot],axis=1)
    #residence
    onehot = pd.get_dummies(train['residence'],prefix='residence')
    train = train.drop('residence',1)
    train = pd.concat([train,onehot],axis=1)
    #house
    onehot = pd.get_dummies(train['house'],prefix='house')
    train = train.drop('house',1)
    train = pd.concat([train,onehot],axis=1)
    if(dataset_num==2):
        #four
        onehot = pd.get_dummies(train['four'],prefix='four')
        train = train.drop('four',1)
        train = pd.concat([train,onehot],axis=1)


    saved = train['target']
    train = train.drop('target',1)
    train = pd.concat([train,saved],axis=1)
    train = train.drop('ID',1)

    for index in list(train):
        if(index=='target'):
            break
        train[index] = train[index].astype(float)

    scaler = StandardScaler()
    scaler.fit(train.drop('target',axis=1))
    scaled_features = scaler.transform(train.drop('target',axis=1))
    df_feat = pd.DataFrame(scaled_features,columns=train.columns[:-1])
    df_feat.head()

    X = df_feat
    y = train['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)

    # 建立 random forest 模型
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest_fit = forest.fit(X_train, y_train)

    # 預測
    test_y_predicted = forest.predict(X_test)

    #print(confusion_matrix(y_test,test_y_predicted)) 
    print(classification_report(y_test,test_y_predicted))

    
train = pd.read_csv('dataset1.csv')
print('Random Forest --- dataset1')
RF(train, 1)


train = pd.read_csv('dataset2.csv')
print('Random Forest --- dataset2')
RF(train, 2)
