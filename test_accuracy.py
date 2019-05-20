import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import joblib
import re
from sklearn.naive_bayes import GaussianNB
import numpy as np
vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('model1.pkl')
from testingmovie import output1 

def dummy_fun(doc):
    return doc

inputCSVfile =   "IMDB.csv"
# reading csv file
print('Reading file:',inputCSVfile)
try:
    my_data = pd.read_csv(inputCSVfile)
except FileNotFoundError:
    print('Error!\n',inputCSVfile,'doesnt exist.')
    exit()
else:
    print("done")


plot_data = my_data['Description']
#print(plot_data[1])

def text_preprocessor(text_data):
    # Remove regular expressions and numbers
    contents = re.sub(r'[\W]', ' ', text_data)
    contents = re.sub("\d+", "", contents)

    # Remove short words
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    contents = shortword.sub('', contents)

    # Tokenization
    txt_tokenized = word_tokenize(contents)
    # print(txt_tokenized)

    # POS tagging - Removing proper nouns only
    txt_pos = [token for token, pos in pos_tag(txt_tokenized) if not pos.startswith('NNP')]
    # print(pos_tag(txt_tokenized))
    # print(txt_pos)

    # Stop words elimination
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in txt_pos if not w in stop_words]

    # Stemming
    ps = PorterStemmer()
    stemmed_out = [ps.stem(w) for w in filtered_sentence]
    
    # print(filtered_sentence)
    return stemmed_out

from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score



pred = [[] for _ in range(99)]
for i in range(99):   
    my_plot = plot_data[i] 
    with open("genres_files.txt", "r") as file:
        my_genres = eval(file.readline())
    contents =my_plot
    contents = contents.strip()
    data = text_preprocessor(contents)
    data = " ".join(data)
    X = vectorizer.transform([data])
    X1 = X.todense()
    y_pred= selector.predict(X1)
    pred[i] = y_pred.toarray().tolist()
#output=pred.tolist()
output2 = [] 
# function used for removing nested  
# lists in python.  
def removeNestings(l): 
    for i in l: 
        if type(i) == list: 
            removeNestings(i) 
        else: 
            output2.append(i) 
  
removeNestings(pred) 
#print(output2)

#Accuracy for Testing
def overlapping_percentage(x, y):
    return (100.0 * len(set(x) & set(y))) / len(set(x) | set(y))
print("accuracy")
#print(output1)
print("Testing data accuracy")
#print(overlapping_percentage(output1,output2))
#print(metrics.accuracy_score(output1,output2))

#result = list(np.where(pred == 1)[1])
#print('\n\nPrediction:')
#for r in result:
#    print('\t*',my_genres[r])
#print(len(output1))
#print(len(output2))
len_set=len(output2)
count1=0
for i in range (len_set):
    if output1[i]==1:
        count1=count1+1
count2=0
for i in range (len_set):
    if output2[i]==output1[i]==1:
        count2=count2+1
print(count2/count1*100)
