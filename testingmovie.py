import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import numpy as np
import re
import os
from tqdm import tqdm
import joblib
from tkinter import *


def clear_folders(my_path):
    if len(os.listdir(my_path) ) == 0:
        print('Empty:',my_path)
    else:
        print('Clearing:',my_path)
        filesToRemove = [os.path.join(my_path,f) for f in os.listdir(my_path)]
        for f in filesToRemove:
            os.remove(f)
    return


my_path = os.getcwd()

clear_folders(os.path.join(my_path,'valuescopy'))

inputCSVfile =   "IMDBforlabel.csv"
# reading csv file
print('Reading file:',inputCSVfile)
try:
    my_data = pd.read_csv(inputCSVfile)
except FileNotFoundError:
    print('Error!\n',inputCSVfile,'doesnt exist.')
    exit()
else:
    print("done")

available_fields = my_data.columns.values.tolist()

num_fields = len(available_fields)
num_values = len(my_data['Rank'])



genres = my_data['Genre']
genre_list = list(genres)

genre_all = []
for my_genre in genre_list:
    g = my_genre.split(',')
    g_stripped = [x.strip() for x in g]# remove white space
    genre_all.extend(g_stripped)

print('\nDetecting unique genre names.')

genre_unique = list(set(genre_all))
genre_unique = sorted(genre_unique)


num_genre = len(genre_unique)


###########
print('Creating label matrix.',end='\t')
y = [[] for _ in range(num_values)]
for i2 in range(num_values):
    for g2 in genre_unique:
        if (genres[i2].find(g2)!= -1):
            y[i2].append(1)
        else:
            y[i2].append(0)
print('Done.\n')
#print(y)
testing=y[1000:1099]

# output list 
output1 = [] 
  
# function used for removing nested  
# lists in python.  
def removeNestings(l): 
    for i in l: 
        if type(i) == list: 
            removeNestings(i) 
        else: 
            output1.append(i) 
  
removeNestings(testing) 
print("\n\n")
#print(output1)

