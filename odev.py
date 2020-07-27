# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:20:38 2020

@author: CemhanSenol
"""

import numpy as np
import pandas as pd

data = pd.read_csv('diabetes.csv')
x_data = data.drop(['Outcome'],axis=1)
y=data.Outcome.values

x= (x_data-np.min(x_data)) / (np.max(x_data)-np.min(x_data))
#%% Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#%% Gausian Naive Bayes
print("BAYES ALGORITMALERI ILE")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
print("Gausian Naive Bayes  ile accuracy : ", gnb.score(x_test,y_test) )
#%% Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
print("Multinomial Naive Bayes ile accuracy Score : ", mnb.score(x_test,y_test) )
#%% Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train,y_train)
print("Bernoulli Naive Bayes ile accuracy : ", bnb.score(x_test,y_test))
#%%KNN 
print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
print("K-Nearest Neighbour(KNN) ALGORITMASI ILE  ")
from sklearn.neighbors import KNeighborsClassifier
score_list = []
each_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
for each in range (1,15):
    knn = KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,y_train)
    score_list=(knn.score(x_test,y_test))
    print("KNN de {} komsu alinirsa accuracy : {} ".format(each,score_list))
#%%Radius Neighbors Classifier
print("o-o--o-o-o-o-o-o-o-o-o-o--o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
from sklearn.neighbors import RadiusNeighborsClassifier
rnn = RadiusNeighborsClassifier()
rnn.fit(x_train,y_train)
print("RNN Algoritmasi kullanilirrsa accuracy :{} ".format(rnn.score(x_test,y_test)))
#%%Logistic Regression
print("o-o--o-o-o-o-o-o-o-o-o-o--o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train)
print("Logistic Regression parametreleri default ise accuracy : ", lr.score(x_test,y_test)) 
#%%Logistic Regression (max_iter=500,sover=saga,penalty=elasticnet)
lr= LogisticRegression(max_iter=500,solver='saga',penalty='elasticnet')
lr.fit(x_train,y_train)
print("Logistic Regression parametreleri degistirip bakalım accuracy : ", lr.score(x_test,y_test))
#%%Logistic Regression CV(Logistic Regression Cross-Validation)
from sklearn.linear_model import LogisticRegressionCV
lrcv = LogisticRegressionCV()
lrcv.fit(x_train,y_train)
print("Logistic RegressionCV ile accuracy : ",lrcv.score(x_test,y_test))
#%%Stochastic  Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier(max_iter=1000)
sgdc.fit(x_train,y_train)
print("Stokastik Grdient Descent ile accuracy : ",sgdc.score(x_test,y_test))
#%%Support Vector Machine(SVM)
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)
print("Support Vector Machine ile accuracy ", svm.score(x_test,y_test))
#%%Decision Tree 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree ile accuracy : ", dt.score(x_test,y_test))
#%%Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=50)
rf.fit(x_train,y_train)
print("Random Forest ile accuracy",rf.score(x_test,y_test))
#%%Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbcs = GradientBoostingClassifier()
gbcs.fit(x_train,y_train)
print("Gradient Bossting ile accuracy : ", gbcs.score(x_test,y_test))
#%%Ada-Boost
from sklearn.ensemble import AdaBoostClassifier
adabc = AdaBoostClassifier()
adabc.fit(x_train,y_train)
print("Ada Boost ile accuracy",adabc.score(x_test,y_test))