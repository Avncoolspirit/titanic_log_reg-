#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:33:23 2019

@author: avnish
"""

import pandas as pd
import numpy as np
import csv
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#import data into a dataframe
df = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#drop values that are irrelavant. The reasonings given in the write up
Filter = df.drop(columns=['Name','Ticket','Cabin'])
test_set = test_data.drop(columns=['Name','Ticket','Cabin'])

#Replace the NaN values with the mean
mean_value = Filter['Age'].mean()
Filter['Age']=Filter['Age'].fillna(mean_value)
Filter= Filter.dropna()

#Replace the NaN Age and Fare values in the test data set by the mean
test_set['Age']=test_set['Age'].fillna(test_set['Age'].mean())
test_set['Fare']=test_set['Fare'].fillna(test_set['Fare'].mean())
test_set = test_set.dropna()

#Make the columns for each option of the Features like SEx, Pclass and Embarked port
embarked = pd.get_dummies(Filter["Embarked"])
sex = pd.get_dummies(Filter["Sex"],drop_first=True)
pClass = pd.get_dummies(Filter["Pclass"]) 
Filter = pd.concat([Filter,embarked,sex,pClass],axis=1)


#Make the columns for each option of the Features like SEx, Pclass and Embarked port
embarked_test = pd.get_dummies(test_set["Embarked"])
sex_test = pd.get_dummies(test_set["Sex"],drop_first=True)
pClass_test = pd.get_dummies(test_set["Pclass"]) 
test_set = pd.concat([test_set,embarked_test,sex_test,pClass_test],axis=1)

#seperate the dependednt variable for training
YTrain =  Filter['Survived']

# prepare the feature set for each observation
Filter = Filter.drop(columns=['Embarked','PassengerId','Pclass','Sex','Survived'])
test_set = test_set.drop(columns=['Embarked','PassengerId','Pclass','Sex','Survived'])


# train the model
logmodel = LogisticRegression()
logmodel.fit(Filter,YTrain)

# make predictions on the test data set
predictions = logmodel.predict(test_set)
predictions_conditioned =  zip(test_data['PassengerId'],predictions)

#Save the predictions in a csv file in the format submitable in Kaggle
with open('gender_submission.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['PassengerID','Survived'])
    for row in predictions_conditioned:
        csv_out.writerow(row)
#print(classification_report(YTest, predictions))
