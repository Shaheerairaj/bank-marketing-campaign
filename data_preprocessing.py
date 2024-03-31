# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:02:08 2024

@author: FR6201
"""

import pandas as pd

data_raw = pd.read_csv("Data/bank-full.csv", delimiter=";")

# Getting the month digit
month_to_numeric = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# Create a new column "month_num" based on the mapping
data_raw['month_num'] = data_raw['month'].map(month_to_numeric)

# Removing outliers from balance and duration
data = data_raw.copy()
for i in ['balance','duration']:
    Q1 = data[i].describe()[4]
    Q3 = data[i].describe()[6]
    IQR = Q3-Q1
    data = data[(data[i] > Q1 - 1.5 * IQR) & (data[i] < Q3 + 1.5*IQR)]

# Map all columns with yes no to binary
map_yes_no = {'yes':1, 'no':0}
for i in ['default','housing','loan','y']:
    data[i] = data[i].map(map_yes_no)

# Dropping duration column since according to the UCI website, 
# this variable highly effects the target, plus if a call is made
# then y is known. Also dropping month categorical column
data.drop(['duration','month'], axis=1, inplace=True)

# Replacing -1 in pdays with 0
data['pdays'].replace(-1, 0, inplace=True)

# Education is ordinal so it won't make sense to one-hot encode it.
map_edu = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}
data['education'] = data['education'].map(map_edu)

# One-hot encoding categorical variables
data = pd.get_dummies(data, columns = ['job','marital','contact','poutcome'])

# Dropping one column for each of the one-hot encoded variables
data.drop(['job_unknown','marital_single','contact_unknown','poutcome_unknown'], axis=1, inplace=True)

data.to_csv('Data/bank-full-preprocessed.csv',index=False)
