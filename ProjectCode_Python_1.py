# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:27:43 2019
@author: eamiagr

"""

## Importing the packages
# Data processing packages
import numpy as np 
import pandas as pd 

# Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

#Import Employee Attrition data
data=pd.read_csv("C:/Users/HR-Employee-Attrition.csv")
data.shape
data.head()

# Check and remediate if there are any null values
data.info()
data.describe()

# Basic statistics of categorical features
data.describe(include=[np.object])
data.columns

#cleaning of data: that not requred
data.drop('EmployeeNumber', axis = 1, inplace = True)
data.drop('Over18', axis = 1, inplace = True)
data.drop('StandardHours', axis = 1, inplace = True)
data.drop('EmployeeCount', axis =1, inplace = True)
data.drop('JobRole', axis =1, inplace = True)
data.shape

data.columns
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)

data.describe(include=[np.object]).columns
## ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

data['EducationField'].unique()
## ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
data['EducationField'].replace('Life Sciences',1, inplace=True)
data['EducationField'].replace('Medical',2, inplace=True)
data['EducationField'].replace('Marketing', 3, inplace=True)
data['EducationField'].replace('Other',4, inplace=True)
data['EducationField'].replace('Technical Degree',5, inplace=True)
data['EducationField'].replace('Human Resources', 6, inplace=True)

data['BusinessTravel'].unique()
## ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
data['BusinessTravel'].replace('Travel_Rarely',1, inplace=True)
data['BusinessTravel'].replace('Travel_Frequently',2, inplace=True)
data['BusinessTravel'].replace('Non-Travel',3, inplace=True)

data['Department'].unique()
## ['Sales', 'Research & Development', 'Human Resources']
data['Department'].replace('Research & Development',1, inplace=True)
data['Department'].replace('Sales',2, inplace=True)
data['Department'].replace('Human Resources', 3, inplace=True)

data['Gender'].unique()
## ['Female', 'Male']
data['Gender'].replace('Male',1, inplace=True)
data['Gender'].replace('Female',0, inplace=True)

data['MaritalStatus'].unique()
## ['Single', 'Married', 'Divorced']
data['MaritalStatus'].replace('Single',1, inplace=True)
data['MaritalStatus'].replace('Married',2, inplace=True)
data['MaritalStatus'].replace('Divorced',0, inplace=True)

data['OverTime'].unique()
## ['Yes', 'No']
data['OverTime'].replace('Yes',1, inplace=True)
data['OverTime'].replace('No',0, inplace=True)

data.describe(include=[np.object]).columns

data.shape
data.columns
data.info()
data.describe()
data.select_dtypes(['object'])
data.head()

X = data.drop('Attrition', axis=1)
y = data['Attrition']

X.columns

 
