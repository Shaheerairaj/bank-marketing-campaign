# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:39:44 2024

@author: FR6201
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Train test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train[:,:8] = sc.fit_transform(X_train[:,:8])
X_test[:,:8] = sc.transform(X_test[:,:8])


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

yhat = model.predict(X3_test)
prediction = list(map(round, yhat))

accuracy_score(y3_test, prediction)

# Linear Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

metrics_lr_acc = accuracy_score(y_test, y_pred)
metrics_lr_recall = recall_score(y_test, y_pred)
metrics_lr_prec = precision_score(y_test, y_pred)
metrics_lr_conf_matrix = confusion_matrix(y_test, y_pred)


# SVM C-Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

metrics_svc_acc = accuracy_score(y_test, y_pred)
metrics_svc_recall = recall_score(y_test, y_pred)
metrics_svc_prec = precision_score(y_test, y_pred)
metrics_svc_conf_matrix = confusion_matrix(y_test, y_pred)


# SVM Linear SVC
iterations = [x for x in range(100,1001,100)]
svm_acc = []
for n in iterations:
    lsvc = LinearSVC(max_iter = n, random_state=0)
    lsvc.fit(X_train, y_train)
    y_pred = lsvc.predict(X_test)
    svm_acc.append(accuracy_score(y_test, y_pred))

plt.plot(iterations, svm_acc)
# max_iter of 400 gives the best results

lsvc = LinearSVC(max_iter = n, random_state=0)
lsvc.fit(X_train, y_train)
y_pred = lsvc.predict(X_test)
svm_acc.append(accuracy_score(y_test, y_pred))

metrics_lsvc_acc = accuracy_score(y_test, y_pred)
metrics_lsvc_recall = recall_score(y_test, y_pred)
metrics_lsvc_prec = precision_score(y_test, y_pred)
metrics_lsvc_conf_matrix = confusion_matrix(y_test, y_pred)


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

metrics_dt_acc = accuracy_score(y_test, y_pred)
metrics_dt_recall = recall_score(y_test, y_pred)
metrics_dt_prec = precision_score(y_test, y_pred)
metrics_dt_conf_matrix = confusion_matrix(y_test, y_pred)


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

metrics_rf_acc = accuracy_score(y_test, y_pred)
metrics_rf_recall = recall_score(y_test, y_pred)
metrics_rf_prec = precision_score(y_test, y_pred)
metrics_rf_conf_matrix = confusion_matrix(y_test, y_pred)

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))



