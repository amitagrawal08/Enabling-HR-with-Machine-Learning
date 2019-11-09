# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:06:55 2019

@author: eamiagr
"""

#print (1)

#import numpy
#import pandas
#import cv2
#
#
#im_g=cv2.imread("smallgray.png",0) ## 0 means grey and 1 means BGR Blue,Green,Red
#im_g

from tkinter import *

window=Tk()

def km_to_miles():
    print(e1_value.get())
    miles=float(e1_value.get())*1.6
    t1.insert(,miles)
    

b1=Button(window,text="Execute", command=km_to_miles)
b1.grid(row=0,column=0)

e1_value=StringVar()
e1=Entry(window,textvariable=e1_value)
e1.grid(row=0,column=1)

t1=Text(window,height=1,width=20)
t1.grid(row=0,column=2)


window.mainloop()