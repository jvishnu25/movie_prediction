#!/usr/bin/python3

from tkinter import *
from movie5_1 import *
import os
import sys
import subprocess

def save():
	text = textPlot.get()
	#print("First Name: %s" % text)
	file = open("endgame_1.txt", "w+")
	file.write(text)
        ##run other program

def run():
	os.system('python3 movie5_1.py')	
	filename = "output_1.txt"  ####
	fo=open(filename,"r")
	text= fo.read()
	label = Label(bom_frame, text= text, font=('century', 8, 'italic', 'bold'), justify='c', bg='lavender').grid(row=3, columnspan=4)
	#3for i in fo:
        	#label = Label(bom_frame, text= str(i), font=('century', 12, 'italic', 'bold'), justify='c', bg='lavender').grid(row=1, columnspan=4)

master = Tk()
master.title("Movie genres Identification")
top_frame = Frame(master, relief=SUNKEN, bg='lavender')
top_frame.pack(fill=X, padx=3, pady=1)
Label(top_frame, text="Movie Genres Identification", font=('century', 14, 'italic', 'bold'), justify='c', bg='lavender').grid(row=0, columnspan=3)
Label(top_frame, text="Enter Text/Plot", font=('century', 10, 'italic', 'bold'), justify='c', bg='lavender').grid(row=1)

textPlot = Entry(top_frame)
textPlot.grid(row=1, column=1)

#Button(top_frame, text='Quit', command=master.quit).grid(row=3, column=1, sticky=W, pady=4)
Button(top_frame, text='Submit', command=save).grid(row=3, column=1, sticky=NS, pady=4)

center_frame = Frame(master, bg='lightblue', height=2, relief=SUNKEN)
center_frame.pack(fill=X, pady=1)
bom_frame = Frame(master, bg='lavender', relief=SUNKEN)
bom_frame.pack(fill=X, padx=3, pady=1)

#Label(bom_frame, text="Result", font=('century', 10, 'italic', 'bold'), justify='c', bg='lavender').grid(row=0, columnspan=4)
	
Button(bom_frame, text='Result', command=run).grid(row=1, column=2, sticky=W, pady=4)

mainloop( )
