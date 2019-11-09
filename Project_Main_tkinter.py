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

#data.describe(include=[np.object]).columns

data.shape
data.columns
data.info()
data.describe()
data.select_dtypes(['object'])
data.head()

X = data.drop('Attrition', axis=1)
y = data['Attrition']

X.columns

X = data[['Age','DistanceFromHome','EnvironmentSatisfaction','Gender','HourlyRate','MaritalStatus','MonthlyRate','PercentSalaryHike','YearsInCurrentRole']]
X.shape
X.columns

X['Age'].unique()
X['DistanceFromHome'].unique()
X['EnvironmentSatisfaction'].unique()   # 1,2,3,4
X['Gender'].unique()                    # 0,1
X['HourlyRate'].unique()
X['MaritalStatus'].unique()             # 1,2,0
X['MonthlyRate'].unique()
X['PercentSalaryHike'].unique()
X['YearsInCurrentRole'].unique()


# Scaling the data values to standardize the range of independent variables
#Feature scaling is a method used to standardize the range of independent variables or features of data.
#Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

# Split the data into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)

def train_test_ml_model(X_train,y_train,X_test,Model):
    model.fit(X_train,y_train) #Train the Model
    y_pred = model.predict(X_test) #Use the Model for prediction

    # Test the Model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_pred)
    accuracy = round(100*np.trace(cm)/np.sum(cm),1)

    #Plot/Display the results
    cm_plot(cm,Model)
    print('Accuracy of the Model' ,Model, str(accuracy)+'%')

#Function to plot Confusion Matrix
def cm_plot(cm,Model):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Comparison of Prediction Result for '+ Model)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
    

# PERFORM PREDICTIONS USING MACHINE LEARNING ALGORITHMS


from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model
Model = "LogisticRegression"
model=LogisticRegression()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis #Import packages related to Model
Model = "LinearDiscriminantAnalysis"
model=LinearDiscriminantAnalysis()

train_test_ml_model(X_train,y_train,X_test,Model)


import pickle 
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(model) 
# Load the pickled model 
model_from_pickle = pickle.loads(saved_model) 
# Use the loaded pickled model to make predictions 
#model_from_pickle.predict(X_test)

X_test.dtype
#from sklearn.externals import joblib 
## Save the model as a pickle in a file 
#joblib.dump(model, 'filename.pkl') 
## Load the model from the file 
#model_from_joblib = joblib.load('filename.pkl')  
## Use the loaded model to make predictions 
#model_from_joblib.predict(Z)


import tkinter as tk
from tkinter import *
import numpy as np

# To create a main window, tkinter offers a method 
# ‘Tk(screenName=None,  baseName=None,  className=’Tk’,  useTk=1)’.
main=tk.Tk() ## where main is the name of the main window object
main.title('Employee Attrition') 
main.geometry("600x400")

def function2():
    A = input1.get()
    B = input2.get()
    C = input3.get()
    D = input4.get()
    E = input5.get()
    F = input6.get()
    G = input7.get()
    H = input8.get()
    I = input9.get()
    Z={'Age' : A,'DistanceFromHome' : B,'EnvironmentSatisfaction' : C,'Gender' : D,'HourlyRate' : E,'MaritalStatus' : F,'MonthlyRate' : G,'PercentSalaryHike' : H,'YearsInCurrentRole' : I}
    Z1= pd.DataFrame(Z, index=[0])
    Value=model_from_pickle.predict(Z1)
    return Value

def function1():
    out = tk.Text(master=main, height=5, width=20)
    out.grid(row=14, column=3)
    out.insert(tk.END, function2())

Label1 = tk.Label(main, text='Age')
Label1.grid(row=2)
Label2 = tk.Label(main, text='DistanceFromHome (KM)')
Label2.grid(row=3)
Label3 = tk.Label(main, text='EnvironmentSatisfaction')
Label3.grid(row=4)
Label31 = tk.Label(main, text='(1-Low, 2-Medium, 3-High, 4-Very High)')
Label31.grid(row=4, column=4)
Label4 = tk.Label(main, text='Gender')
Label4.grid(row=5)
Label41 = tk.Label(main, text='(1-Male, 0-Female)')
Label41.grid(row=5, column=4)
Label5 = tk.Label(main, text='HourlyRate')
Label5.grid(row=6)
Label6 = tk.Label(main, text='MaritalStatus')
Label6.grid(row=7)
Label61 = tk.Label(main, text='(1-Single, 2-Married, 0-Divorcee)')
Label61.grid(row=7, column=4)
Label7 = tk.Label(main, text='MonthlyRate')
Label7.grid(row=8)
Label8 = tk.Label(main, text='PercentSalaryHike')
Label8.grid(row=9)
Label9 = tk.Label(main, text='YearsInCurrentRole')
Label9.grid(row=10)

# EnvironmentSatisfaction 1-'Low', 2-'Medium', 3-'High', 4-  'Very High'
# Gender 1-'Male', 0-'Female'
# MaritalStatus 1-'Single', 2-'Married', 0-'Divorcee'


input1 = tk.Entry()
input1.grid(row=2, column=3)
input2 = tk.Entry()
input2.grid(row=3, column=3) 
input3 = tk.Entry()
input3.grid(row=4, column=3) 
input4 = tk.Entry()
input4.grid(row=5, column=3) 
input5 = tk.Entry()
input5.grid(row=6, column=3) 
input6 = tk.Entry()
input6.grid(row=7, column=3) 
input7 = tk.Entry()
input7.grid(row=8, column=3)
input8 = tk.Entry()
input8.grid(row=9, column=3) 
input9 = tk.Entry()
input9.grid(row=10, column=3) 

A = input1.get()
B = input2.get()
C = input3.get()
D = input4.get()
E = input5.get()
F = input6.get()
G = input7.get()
H = input8.get()
I = input9.get()
#Z=pd.Series([A,B,C,D,E,F,G,H,I])
 
Z2={'Age' : A,'DistanceFromHome' : B,'EnvironmentSatisfaction' : C,'Gender' : D,'HourlyRate' : E,'MaritalStatus' : F,'MonthlyRate' : G,'PercentSalaryHike' : H,'YearsInCurrentRole' : I}
Z3= pd.DataFrame(Z2, index=[0])
#model_from_pickle.predict(Z1)


button = tk.Button(text='Submit', width=15, command=function2)
button.grid(row=11, column=3)

main.mainloop()