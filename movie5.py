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

clear_folders(os.path.join(my_path,'values'))

inputCSVfile =   "IMDB-Movie-Data (1).csv"
# reading csv file
print('Reading file:',inputCSVfile)
try:
    my_data = pd.read_csv(inputCSVfile)
except FileNotFoundError:
    print('Error!\n',inputCSVfile,'doesnt exist.')
    exit()
else:
    print(my_data.head())

available_fields = my_data.columns.values.tolist()

num_fields = len(available_fields)
num_values = len(my_data['Rank'])

print('\nTotal number of fields:',num_fields)

for count,f in enumerate(available_fields):
    print(count+1,f)

print(num_values,' values.')

#### Preprocessing
print('\nChecking for empty fields in ',inputCSVfile,end='.\n')
for f in available_fields:
    s = my_data[f].isnull().sum()
    print('Checking ',f,end='\t')
    if s == 0:
        #No missing fields
        print('OK')
    else:
        print('ERROR.',s,' values missing')
        #print('Removing Empty rows')
        #my_data = my_data.dropna()
  
print('\nFetching genre names.')

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


for my_genre in genre_unique:
    print('\t* ',my_genre)
num_genre = len(genre_unique)
print('Number of genres:',num_genre)

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
###########
print('\nCreating noun-verb dictionaries.')
for g in genre_unique:
    
    gv = g + '_verb.txt'
    gn = g + '_noun.txt'
    
    pathV = os.path.join(my_path,'values',gv)
    pathN = os.path.join(my_path,'values',gn)
    
    fV= open(pathV,"w+")
    fN= open(pathN,"w+")

def text_preprocessor(text_data,word_type):
    # Remove regular expressions and numbers
    contents = re.sub(r'[\W]', ' ', text_data)
    contents = re.sub("\d+", "", contents)

    # Remove short words
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    contents = shortword.sub('', contents)

    # Tokenization
    txt_tokenized = word_tokenize(contents)
    # print(txt_tokenized)
    if word_type == 'verb':
        # POS tagging 
        txt_pos = [token for token, pos in pos_tag(txt_tokenized) if pos.startswith('V')]
    elif word_type == 'noun':
        # POS tagging 
        txt_pos = [token for token, pos in pos_tag(txt_tokenized) if pos.startswith('N')]
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

plot_data = my_data['Description']
#feat_all = []
for i in range(num_values):
    
    my_plot = plot_data[i]
    my_title = my_data['Title'].iloc[i]
    my_genres = genre_list[i]
    
    g = my_genres.split(',')
    g_stripped = [x.strip() for x in g]# remove white space
    
    featN = text_preprocessor(my_plot,'noun')
    featV = text_preprocessor(my_plot,'verb')

    fN = '\n'.join(featN)
    fV = '\n'.join(featV)
    
    # Creating dictionary
    for gg in g_stripped:

        fileN_to_open = os.path.join(my_path,'values',(gg + '_noun.txt'))
        fileV_to_open = os.path.join(my_path,'values',(gg + '_verb.txt'))

        fileN = open(fileN_to_open,'a+')
        fileN.write("\n")
        fileN.write(fN)

        fileV = open(fileV_to_open,'a+')
        fileV.write("\n")
        fileV.write(fV)
        
        if gg == 'Fantasy':
            #print(fN,fV)
            a= 0
    if i<5:#displaying some values.
        print('\n\n',my_title.upper(),end=':\n')
        for f in featN:
            print(f,end='  ')
            
        for f in featV:
            print(f,end='  ')
            
    #feat_all.append(featN)

'''
# Select a genre to see its word cloud
my_genre = 'Sport'
print('\nShowing WordCloud for:\n\t\t\t',my_genre)
to_show = []
for n in range(num_values):
    g = genres[n]
    if my_genre in g.split(','):
        to_show.append(corpus[n])

to_show = " ".join(to_show)

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

wc = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10)
wordcloud = wc.generate(to_show) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()'''
#########################################
import os


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


fullname = ""
my_path = os.getcwd()
print('\nReading from saved data:')
data_dir = os.path.join(my_path,'values')
onlyfiles = [f for f in os.listdir(data_dir)]
num_files = len(onlyfiles)
wordSet = set([])
wordDicts = []
bow_vals = []

for i3,f3 in enumerate(onlyfiles):
    print('Fetching data from ',f3)
    fullname = os.path.join(data_dir,f3)
    f = open(fullname, "r")
    contents =f.read()
    doc = contents.strip()

    bow = doc.split("\n")
    w = ""
    if w in bow:
        bow.remove("")
        
    bow_vals.append(bow)
    
    for b in bow:
        wordSet.add(b)
    info = doc.split('\n')


    wordDicts.append(dict.fromkeys(wordSet, 0))

    for word in bow:
        wordDicts[i3][word]+=1

       # print(wordDicts)
import pandas as pd
pd.DataFrame(wordDicts)

tf_idf_vals = []
tf_vals = []
idf_vals = []

for i4 in range(num_files):
    tfBow = computeTF(wordDicts[i4], bow_vals[i4])
    tf_vals.append(tfBow)

print('\nFinding unique words...')
words_uniq = set([])
for tf in tf_vals:
    for k in tf.keys():
        words_uniq.add(k)

words_uniq = list(words_uniq)
print('\nCreating feature matrix from dictionary...')
feat_all = [[] for _ in range(num_values)]
#print(feat_all)
for i4 in (range(num_values)):
    #print(i4,end=' ')
    my_plot = plot_data[i4]
    featN = text_preprocessor(my_plot,'noun')
    featV = text_preprocessor(my_plot,'verb')
    fff = 0
    f5 = []
    flag = 0
    for w in words_uniq:
        flag = 0
        if w in featN:
        #for f_n in featN:
        #if f_n in words_uniq:            
            for my_dict in tf_vals:
                if w in my_dict.keys():
                    result = my_dict[w]
                    feat_all[i4].append(result)
                    break
            '''#k = my_dict.keys()
            #v = my_dict.values()
                for k,v in my_dict.items():
                    if w == k :
                        #fff = fff + 1 
                        result = my_dict[w]
                        f5.append(result)
                        break
                        flag = 1
                    #else:
                        #f5.append(0)
                if flag == 1:
                    break'''
        else :
            feat_all[i4].append(0)
    #feat_all[i4] = f5
#print(feat_all)
    #print(fff)

       # print(y)        

X = feat_all

print('Performing classification.',end='\t')



from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())
X = np.array(X)
y = np.array(y)
# train
classifier.fit(X, y)

# predict
predictions = classifier.predict(X[1])

pred = predictions.toarray()
result = list(np.where(pred == 1)[1])
print('\n\nPrediction:')
for r in result:
    print('\t*',genre_unique[r])

#joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(classifier, 'model.pkl')

with open("genres_files.txt", "w") as file:
    file.write(str(genre_unique))

with open("words_file.txt", "w") as file:
    file.write(str(words_uniq))
#print(metrics.accuracy_score(y,predictions))


################################################
    ###############################################


print('\n\n\t****Single File Test****\n')

filename = "endgame.txt"
f=open(filename,"r")
print('Input File:\n\t',filename)

with open("genres_files.txt", "r") as file:
    my_genres = eval(file.readline())

with open("words_file.txt", "r") as file:
    words_uniq = eval(file.readline())
    
contents =f.read()
contents = contents.strip()

#data = text_preprocessor(contents,'noun')
#data = " ".join(data)
#############################

featN = text_preprocessor(contents,'noun')
fff = 0
f5 = []
flag = 0
feat_all = []
for w in words_uniq:
    flag = 0
    if w in featN:
    #for f_n in featN:
    #if f_n in words_uniq:            
        for my_dict in tf_vals:
            if w in my_dict.keys():
                result = my_dict[w]
                feat_all.append(result)
                break
        '''#k = my_dict.keys()
        #v = my_dict.values()
            for k,v in my_dict.items():
                if w == k :
                    #fff = fff + 1 
                    result = my_dict[w]
                    f5.append(result)
                    break
                    flag = 1
                #else:
                    #f5.append(0)
            if flag == 1:
                break'''
    else :
        feat_all.append(0)

#############################
X = np.array(feat_all)

#X1 = X.todense()

y_pred= classifier.predict(X)
pred = y_pred.toarray()
result = list(np.where(pred == 1)[1])
print('\n\nPrediction:')
for r in result:
    print('\t*',my_genres[r])


