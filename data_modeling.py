# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:39:44 2024

@author: FR6201
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def accuracy_scores(models):
    accuracy = []
    recall = []
    precision = []
    f1Score = []
    train_time = []
    
    for model in models:
        start_time = pd.datetime.now()
        model.fit(X_train, y_train)
        end_time = pd.datetime.now()
        
        y_pred = model.predict(X_test)
        
        accuracy.append(round(accuracy_score(y_test, y_pred),4))
        recall.append(round(recall_score(y_test, y_pred),4))
        precision.append(round(precision_score(y_test, y_pred),4))
        f1Score.append(round(f1_score(y_test, y_pred),4))
        train_time.append(end_time - start_time)
    
    return accuracy, recall, precision, f1Score, train_time



data = pd.read_csv('Data/bank-full-preprocessed.csv')

# Re-arranging columns for ease during scaling
y = data.pop('y')
default = data.pop('default')
housing = data.pop('housing')
loan = data.pop('loan')

data['default'] = default
data['housing'] = housing
data['loan'] = loan
data['y'] = y

# Seperating X and y variables
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Separating x and y variables for sm models
y2 = data[['y']]
X2 = data.drop('y', axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train test split for stats models
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# Stats models linear regression
X2_sm = X2_train = sm.add_constant(X2_train)
model = sm.OLS(y2_train, X2_sm).fit()
model.summary()

# Stats model without the less significant variables
X3 = data.drop(['age','pdays','job_admin.','job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_self-employed','job_services','job_technician','job_unemployed','default','y'], axis=1)

# Train test split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y2, test_size=0.2, random_state=0)

X3_sm = X3_train = sm.add_constant(X3_train)
model = sm.OLS(y3_train, X3_sm).fit()
model.summary()


# Feature Scaling
sc = StandardScaler()
X_train[:,:8] = sc.fit_transform(X_train[:,:8])
X_test[:,:8] = sc.transform(X_test[:,:8])


# Linear Regression
lr = LogisticRegression()


# SVM C-Support Vector Machines
svc = SVC()


# SVM Linear SVC
iterations = [x for x in range(100,1001,100)]
svm_acc = []
for n in iterations:
    lsvc = LinearSVC(max_iter = n, random_state=0)
    lsvc.fit(X_train, y_train)
    y_pred = lsvc.predict(X_test)
    svm_acc.append(accuracy_score(y_test, y_pred))

plt.plot(iterations, svm_acc)
# max_iter of 300 gives the best results

lsvc = LinearSVC(max_iter = 300, random_state=0)


# KNN Classification
neighbors = [x for x in range(1,11)]
acc = []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))

plt.plot(neighbors, acc)

# n_neighbors tapers off after just 2 which makes sense since it's a binary classification model
knn = KNeighborsClassifier(n_neighbors = 2)


# Decision Tree Classifier
criterion = ['gini','entropy']
dt_acc = []
for c in criterion:
    dt = DecisionTreeClassifier(criterion=c)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_acc.append(accuracy_score(y_test, y_pred))

plt.plot(criterion, dt_acc)

# Entropy seems to give the best results
dt = DecisionTreeClassifier(criterion='entropy')


# Random Forest
estimators = [x for x in range(50, 501, 50)]
rf_acc = []
for e in estimators:
    rf = RandomForestClassifier(n_estimators = e, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc.append(accuracy_score(y_test, y_pred))

plt.plot(estimators, rf_acc)

# n_estimators = 250 seems to give the best results
rf = RandomForestClassifier(n_estimators = 250, criterion='entropy', random_state=0)


# Making an accuracy matrix to measure performance of all models
models = [lr, svc, lsvc, knn, dt, rf]
model_names = ['Linear Regression','SVC','Linear SVM','KNN','Decision Trees','Random Forrest']
accuracy, recall, precision, f1Score, train_time = accuracy_scores(models)
accuracy_matrix = pd.DataFrame({
    'Accuracy':accuracy,
    'Recall':recall,
    'Precision':precision,
    'F1 Score':f1Score,
    'Training Time':train_time},index=model_names)

