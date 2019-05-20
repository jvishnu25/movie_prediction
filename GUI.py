#!/usr/bin/python

from tkinter import * 
from tkinter import messagebox 
import os
import sys


def left1(event):
	text = textPlot.get()
	#print("First Name: %s" % text)
	file = open("endgame_1.txt", "w+")
	file.write(text)
	b2= Button(frame1, text='Result')
	b2.config( bg='gray91',activeforeground = 'dark blue')
	b2.grid(row=6, columnspan =4, padx=30, pady=30)
	b2.bind('<Button-1>', left2) # bind left mouse clicks

def left2(event):
	os.system('python3 movie5_1.py')###os.system('python3 movie5_1.py') ------ if required change it
	filename = "output_1.txt" 
	fo=open(filename,"r")
	text= fo.read()
	l4 = Label(frame1, text= text, font=('Cambria', 18, 'bold'), bg='grey91', justify='c')
	l4.grid(row =8, columnspan =4, padx=30, pady=30)

def clear():
	textPlot.delete(0, 'end')
	
	
	
master = Tk()
master.title("Movie Genre Identification")
frame1 = Frame(master, relief=SUNKEN, bg='light grey', borderwidth=3)
frame1.pack(fill=X, padx=1, pady=1)

l1 = Label(frame1, text="Movie Genre Identification", font=('Cambria', 24, 'italic', 'bold'), justify='c', bg='gray91', fg = 'slate blue').grid(row=1, columnspan=4, padx=30, pady=30)

l2 = Label(frame1, text="Enter Text/Plot :", font=('Arial', 18,  'bold'), justify=LEFT, bg='light grey').grid(row=3,column = 1, padx=30, pady=30)

textPlot = Entry(frame1)
textPlot.grid(row=3, column = 2)

b= Button(frame1, text='Clear',command=clear)
b.config( bg='powder blue',activeforeground = 'dark blue')
b.grid(row=4, column=1, sticky=E, padx=30, pady=30)

b1= Button(frame1, text='Submit')
b1.config( bg='powder blue',activeforeground = 'dark blue')
b1.grid(row=4, column=2, sticky=E, padx=30, pady=30)
b1.bind('<Button-1>', left1) # bind left mouse click

#master.bind_all('<Button-1>',callback) #click on image close window 08-11-2017 whole tk	
mainloop( )
