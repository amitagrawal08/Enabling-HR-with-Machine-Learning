# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:27:43 2019
@author: Amit Agrawal

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

#cleaning of data
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

from sklearn.svm import SVC,NuSVC  #Import packages related to Model
Model = "SVC"
model=SVC() #Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.svm import SVC,NuSVC  #Import packages related to Model
Model = "NuSVC"
model=NuSVC(nu=0.285)#Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


from xgboost import XGBClassifier  #Import packages related to Model
Model = "XGBClassifier()"
model=XGBClassifier() #Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.neighbors import KNeighborsClassifier  #Import packages related to Model
Model = "KNeighborsClassifier"
model=KNeighborsClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.naive_bayes import GaussianNB,MultinomialNB  #Import packages related to Model
Model = "GaussianNB"
model=GaussianNB()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model
Model = "SGDClassifier"
model=SGDClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model
Model = "LogisticRegression"
model=LogisticRegression()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier #Import packages related to Model
Model = "DecisionTreeClassifier"
model=DecisionTreeClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier #Import packages related to Model
Model = "ExtraTreeClassifier"
model=ExtraTreeClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis #Import packages related to Model
Model = "QuadraticDiscriminantAnalysis"
model = QuadraticDiscriminantAnalysis()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis #Import packages related to Model
Model = "LinearDiscriminantAnalysis"
model=LinearDiscriminantAnalysis()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model
Model = "RandomForestClassifier"
model=RandomForestClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model
Model = "AdaBoostClassifier"
model=AdaBoostClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model
Model = "GradientBoostingClassifier"
model=GradientBoostingClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model
Model = "LogisticRegression"
model=LogisticRegression()

train_test_ml_model(X_train,y_train,X_test,Model)

# import pickle package to save the model and use it later
import pickle 
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(model) 
# Load the pickled model 
model_from_pickle = pickle.loads(saved_model) 
# Use the loaded pickled model to make predictions 
#model_from_pickle.predict(X_test)

#X_test.dtype

# import tkinter to make developing graphical user interfaces (GUIs) to show our model prediction
import tkinter as tk
# also import the themed tkinter for good looking widgets (not obligatory)
from tkinter import *
from tkinter import ttk


class Widget:
    def __init__(self):
        window = tk.Tk()
        #window.title("Employee Attrition")
        window.resizable(0, 0) # prohibit resizing the window
        #window.resizable(600x400)

        Label1 = tk.Label(window, text='Age')
        Label1.grid(row=1, column=0, sticky=W)
        Label2 = tk.Label(window, text='DistanceFromHome (KM)')
        Label2.grid(row=2, column=0, sticky=W)
        Label3 = tk.Label(window, text='EnvironmentSatisfaction')
        Label3.grid(row=3, column=0, sticky=W)
        Label4 = tk.Label(window, text='(1-Low, 2-Medium, 3-High, 4-Very High)')
        Label4.grid(row=3, column=2, sticky=W)
        Label5 = tk.Label(window, text='Gender')
        Label5.grid(row=4, column=0, sticky=W)
        Label6 = tk.Label(window, text='(1-Male, 0-Female)')
        Label6.grid(row=4, column=2, sticky=W)
        Label7 = tk.Label(window, text='HourlyRate')
        Label7.grid(row=5, column=0, sticky=W)
        Label8 = tk.Label(window, text='MaritalStatus')
        Label8.grid(row=6, column=0, sticky=W)
        Label9 = tk.Label(window, text='(1-Single, 2-Married, 0-Divorcee)')
        Label9.grid(row=6, column=2, sticky=W)
        Label10 = tk.Label(window, text='MonthlyRate')
        Label10.grid(row=7, column=0, sticky=W)
        Label11 = tk.Label(window, text='PercentSalaryHike')
        Label11.grid(row=8, column=0, sticky=W)
        Label12 = tk.Label(window, text='YearsInCurrentRole')
        Label12.grid(row=9, column=0, sticky=W)
        
        label_result = tk.Label(window, text='Result:')
        label_result.grid(row=13, column=0, sticky=W)
        
        self.label = StringVar() # create an id for the invisible label where will be displayed the text in the box
        invisible_label = tk.Label(window, text='', textvariable=self.label) # create the invisible label
        invisible_label.grid(row=13, column=1, sticky=E)
        
        self.entry1_id = StringVar() # create an id for your entry, this helps getting the text
        entry1 = tk.Entry(window, textvariable=self.entry1_id, justify=LEFT)
        entry1.grid(row=1, column=1, sticky=E)
        self.entry2_id = StringVar() # create an id for your entry, this helps getting the text
        entry2 = tk.Entry(window, textvariable=self.entry2_id, justify=LEFT)
        entry2.grid(row=2, column=1, sticky=E)
        self.entry3_id = StringVar() # create an id for your entry, this helps getting the text
        entry3 = tk.Entry(window, textvariable=self.entry3_id, justify=LEFT)
        entry3.grid(row=3, column=1, sticky=E)
        self.entry4_id = StringVar() # create an id for your entry, this helps getting the text
        entry4 = tk.Entry(window, textvariable=self.entry4_id, justify=LEFT)
        entry4.grid(row=4, column=1, sticky=E)
        self.entry5_id = StringVar() # create an id for your entry, this helps getting the text
        entry5 = tk.Entry(window, textvariable=self.entry5_id, justify=LEFT)
        entry5.grid(row=5, column=1, sticky=E)
        self.entry6_id = StringVar() # create an id for your entry, this helps getting the text
        entry6 = tk.Entry(window, textvariable=self.entry6_id, justify=LEFT)
        entry6.grid(row=6, column=1, sticky=E)
        self.entry7_id = StringVar() # create an id for your entry, this helps getting the text
        entry7 = tk.Entry(window, textvariable=self.entry7_id, justify=LEFT)
        entry7.grid(row=7, column=1, sticky=E)
        self.entry8_id = StringVar() # create an id for your entry, this helps getting the text
        entry8 = tk.Entry(window, textvariable=self.entry8_id, justify=LEFT)
        entry8.grid(row=8, column=1, sticky=E)
        self.entry9_id = StringVar() # create an id for your entry, this helps getting the text
        entry9 = tk.Entry(window, textvariable=self.entry9_id, justify=LEFT)
        entry9.grid(row=9, column=1, sticky=E)
        
        button = tk.Button(window, text='Submit', command=self.clicked)
        button.grid(row=11, column=1, sticky=E)
        window.bind("<Return>", self.clicked) # handle the enter key event of your keyboard
        button.bind("<Button-1>", self.clicked) # bind the action of the left button of your mouse to the button assuming your primary click button is the left one.
        
        window.mainloop() # call the mainloop function so the window won't fade after the first execution
        
    def clicked(self, event):
        A = float(self.entry1_id.get()) # get the text from entry
        B = float(self.entry2_id.get())
        C = float(self.entry3_id.get())
        D = float(self.entry4_id.get())
        E = float(self.entry5_id.get())
        F = float(self.entry6_id.get())
        G = float(self.entry7_id.get())
        H = float(self.entry8_id.get())
        I = float(self.entry9_id.get())
        Z={'Age' : A,'DistanceFromHome' : B,'EnvironmentSatisfaction' : C,'Gender' : D,'HourlyRate' : E,'MaritalStatus' : F,'MonthlyRate' : G,'PercentSalaryHike' : H,'YearsInCurrentRole' : I}
        Z1= pd.DataFrame(Z, index=[0])
        text=model_from_pickle.predict(Z1)
        
        reply = self.label.set(format(text)) # format the text on the invisible label you created above
        return reply

Widget()

