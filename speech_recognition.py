# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:50:30 2019

@author: eamiagr
"""

import speech_recognition as sr

r =  sr.Recognizer()

with sr.Microphone() as source:
    print('Speak Anything : ')
    audio=r.listen(source)
    
    try:
        text= r.recognize_google(audio)
        print('You said: {}'.format(text))
    except:
        print('Sorry, We could not recognize your voice')

