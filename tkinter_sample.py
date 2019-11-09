# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:59:16 2019

@author: eamiagr
"""

import tkinter as tk
import random

window = tk.Tk()
window.title("Tkinter playground")
window.geometry("400x400")
random.seed()
list = ["Give me", "Find me", "Where is", "Why you"]

def sentence_generator():
    sentence = entry.get()
    return list[random.randint(0,3)] + " " + sentence

def display_name():
    text = tk.Text(master=window, height=10, width=50)
    text.grid(row=3, column=1)
    text.insert(tk.END, sentence_generator())

title = tk.Label(text="ENTER YOUR NAME")
title.grid(row=0, column=1)

button = tk.Button(text="Click me!", command=display_name)
button.grid(row=2, column=1)

entry = tk.Entry()
entry.grid(row=1, column = 1)



window.mainloop()