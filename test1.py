import tkinter as tk # also import the themed tkinter for good looking widgets (not obligatory)
from tkinter import *

class Widget:
    def __init__(self):
        window = tk.Tk()
        window.title("Employee Attrition")
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
        text= pd.DataFrame(Z, index=[0])
#        text=A+B
        
        reply = self.label.set(text) # format the text on the invisible label you created above
        return reply

Widget()