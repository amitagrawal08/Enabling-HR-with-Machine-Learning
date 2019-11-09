import tkinter as tk
from tkinter import *
import numpy as np
import pandas as pd

# To create a main window, tkinter offers a method 
# ‘Tk(screenName=None,  baseName=None,  className=’Tk’,  useTk=1)’.
main=tk.Tk() ## where main is the name of the main window object
main.title('Employee Attrition') 
main.geometry("600x400")



def function2():
    entries = []
    A = tk.Entry.get(input1)
    B = float(input2.get())
    C = float(input3.get())
    D = float(input4.get())
    E = float(input5.get())
    F = float(input6.get())
    G = float(input7.get())
    H = float(input8.get())
    I = float(input9.get())
    Z={'Age' : A,'DistanceFromHome' : B,'EnvironmentSatisfaction' : C,'Gender' : D,'HourlyRate' : E,'MaritalStatus' : F,'MonthlyRate' : G,'PercentSalaryHike' : H,'YearsInCurrentRole' : I}
    Z1= pd.DataFrame(Z, index=[0])
#    Label20 = tk.Label(main, text=Z1)
#    Label20.grid(row=15)
    return Z1

def function1():
    out = tk.Text(master=main, height=5, width=20)
    out.grid(row=14, column=3)
    out.insert(tk.END, function2())


#def function2():
#    A = input1.get()
#    B = input2.get()
#    C = input3.get()
#    D = input4.get()
#    E = input5.get()
#    F = input6.get()
#    G = input7.get()
#    H = input8.get()
#    I = input9.get()
#    Z=np.array([A,B,C,D], dtype='float')
#    return model_from_pickle.predict(Z)

#def function1():
#    out = tk.Text(master=main, height=5, width=20)
#    out.grid(row=14, column=3)
#    out.insert(tk.END, function2())

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

input1 = tk.Entry(main)
input1.grid(row=2, column=3)
#input1.pack()
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

A = tk.Entry.get(input1)
B = input2.get()
C = input3.get()
D = input4.get()
E = input5.get()
F = input6.get()
G = input7.get()
H = input8.get()
I = input9.get()
#Z=pd.Series([A,B,C,D,E,F,G,H,I])
entry=[]

Z={'Age' : A,'DistanceFromHome' : B,'EnvironmentSatisfaction' : C,'Gender' : D,'HourlyRate' : E,'MaritalStatus' : F,'MonthlyRate' : G,'PercentSalaryHike' : H,'YearsInCurrentRole' : I}
Z1= pd.DataFrame(Z, index=[0])

button = tk.Button(text='Submit', width=15, command=function1)
button.grid(row=11, column=3)

main.mainloop()




