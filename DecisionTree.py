#!/usr/bin/env python
# coding: utf-8

# In[22]:


################decision_tree
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def DT_gini(train, dataset_num, criterion):
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

    saved=train['target']
    train = train.drop('target',1)
    train = pd.concat([train,saved],axis=1)
    features = list(train)

    target='target'
    ID='ID'
    train['target'].value_counts()

    x_columns0 = [x for x in train.columns if x not in [target, ID]]
    X = train[x_columns0]
    y = train['target']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3,random_state=0)

    if(criterion=='gini'):
        clf = tree.DecisionTreeClassifier()
    else:
        clf = tree.DecisionTreeClassifier(criterion='entropy')
    buy_clf = clf.fit(train_X, train_y)

    test_y_predicted = buy_clf.predict(test_X)


    print(classification_report(test_y,test_y_predicted))

    

train = pd.read_csv('dataset1.csv')
print('Decision Tree --- criterion=gini --- dataset1')
DT_gini(train,1,'gini')


train = pd.read_csv('dataset2.csv')
print('Decision Tree --- criterion=gini --- dataset2')
DT_gini(train,2,'gini')


train = pd.read_csv('dataset1.csv')
print('Decision Tree --- criterion=entropy --- dataset1')
DT_gini(train,1,'entropy')


train = pd.read_csv('dataset2.csv')
print('Decision Tree --- criterion=entropy --- dataset2')
DT_gini(train,2,'entropy')

"""dot_data = tree.export_graphviz(buy_clf, out_file=None, feature_names=features[1:16], class_names=['not_yet','want'],filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())"""







