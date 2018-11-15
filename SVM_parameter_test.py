#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def data(train, dataset_num):
    train = train.drop('ID',1)
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
    
    return X_train, X_test, y_train, y_test
    

def parameter_test(X_train, X_test, y_train, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1,0.01,0.001,0.0001],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision']
    warnings.filterwarnings('ignore')
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()

        print(clf.best_params_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)

        print(classification_report(y_true, y_pred))

        print()
        
        
train = pd.read_csv('dataset1.csv')
X_train, X_test, y_train, y_test = data(train,1)
parameter_test(X_train, X_test, y_train, y_test)

train = pd.read_csv('dataset2.csv')
X_train, X_test, y_train, y_test = data(train,2)
parameter_test(X_train, X_test, y_train, y_test)


# In[ ]:




