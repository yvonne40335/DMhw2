#!/usr/bin/env python
# coding: utf-8

# In[10]:


#######################svm
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def SVM(train, dataset_num, kernel):
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
    
    
    if(kernel=='linear'):
        svclassifier = SVC(kernel='linear')
    else:
        svclassifier = SVC(C=10,gamma=0.01,kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
 
    print(classification_report(y_test,y_pred))

train = pd.read_csv('dataset1.csv')
print('SVM --- linear --- dataset1')
SVM(train, 1, 'linear')


train = pd.read_csv('dataset2.csv')
print('SVM --- linear --- dataset2')
SVM(train, 2, 'linear')


train = pd.read_csv('dataset1.csv')
print('SVM --- rbf --- C=10, gamma=0.01 --- dataset1')
SVM(train, 1, 'rbf')


train = pd.read_csv('dataset2.csv')
print('SVM --- rbf --- C=10, gamma=0.01 --- dataset2')
SVM(train, 2, 'rbf')


# In[ ]:




