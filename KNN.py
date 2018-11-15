#!/usr/bin/env python
# coding: utf-8

# In[6]:


###########if you can use ipython, then you can uncomment the line 'get_ipython().run_line_magic('matplotlib', 'inline')', thus you can see the chart about different k

##########################knn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
#get_ipython().run_line_magic('matplotlib', 'inline')

def KNN(train, dataset_num, k):
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


    X = df_feat
    y = train['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
    
    
    error_rate = []

    for num in range(1,60):  
        knn = KNeighborsClassifier(n_neighbors=num)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))


    #將k=1~60的錯誤率製圖畫出。
    plt.figure(figsize=(10,6))
    plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    if(dataset_num==1):
        plt.title('Error Rate vs. K Value --- dataset1')
    else:
        plt.title('Error Rate vs. K Value --- dataset2')
    plt.xlabel('K')
    plt.ylabel('Error Rate')


    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)


    print(classification_report(y_test,pred))

    
train = pd.read_csv('dataset1.csv')
print('KNN --- k=2 --- dataset1')
KNN(train,1,2)    
    
train = pd.read_csv('dataset2.csv')
print('KNN --- k=20 --- dataset2')
KNN(train,2,20)


# In[ ]:




