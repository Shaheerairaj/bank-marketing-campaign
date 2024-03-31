# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:39:44 2024

@author: FR6201
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

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
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_pred = lsvc.predict(X_test)

metrics_lsvc_acc = accuracy_score(y_test, y_pred)
metrics_lsvc_recall = recall_score(y_test, y_pred)
metrics_lsvc_prec = precision_score(y_test, y_pred)
metrics_lsvc_conf_matrix = confusion_matrix(y_test, y_pred)