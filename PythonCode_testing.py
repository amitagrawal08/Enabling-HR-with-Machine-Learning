# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:28:48 2019

@author: eamiagr
"""

##Importing the packages
#Data processing packages
import numpy as np 
import pandas as pd 

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

#Machine Learning packages
from sklearn.svm import SVC,NuSVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler	
from sklearn.metrics import confusion_matrix

#Suppress warnings
import warnings
warnings.filterwarnings('ignore')

#Import Employee Attrition data
data=pd.read_csv("C:/Users/HR-Employee-Attrition.csv")
data.head()

# Check and remediate if there are any null values¶
data.info()

# Check and remove if there are any fields which does not add value¶
data['Over18'].value_counts()
data.describe()

#These fields does not add value, hence removed
data = data.drop(['EmployeeCount','Over18'], axis = 1)

# Perform datatype conversion or translation wherever required¶
#A lambda function is a small anonymous function.
#A lambda function can take any number of arguments, but can only have one expression.
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)


# Convert Categorical values to Numeric Values¶
data.head()

#This function is used to convert Categorical values to Numerical values
data=pd.get_dummies(data)
data.head()

# Separating Feature and Target matrices
X = data.drop(['Attrition'], axis=1)
y=data['Attrition']

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

#from sklearn.svm import SVC,NuSVC  #Import packages related to Model
#Model = "SVC"
#model=SVC() #Create the Model

#train_test_ml_model(X_train,y_train,X_test,Model)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model
Model = "GradientBoostingClassifier"
model=GradientBoostingClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)

Z = data.head(3).tail(1).drop(['Attrition'], axis=1)

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(model) 
  
# Load the pickled model 
model_from_pickle = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
model_from_pickle.predict(Z) 


from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(model, 'filename.pkl') 
  
# Load the model from the file 
model_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
model_from_joblib.predict(Z)

