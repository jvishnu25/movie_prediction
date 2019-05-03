import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import numpy as np
import re
import joblib

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
for my_genre in genre_unique:
    print('\t* ',my_genre)
num_genre = len(genre_unique)
print('Number of genres:',num_genre)


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

plot_data = my_data['Description']
feat_all = []
for i in range(num_values):
    
    my_plot = plot_data[i]
    my_title = my_data['Title'].iloc[i]
    feat = text_preprocessor(my_plot)
    
    if i<5:#displaying some values.
        print('\n\n',my_title.upper(),end=':\n')
        for f in feat:
            print(f,end='  ')
    feat_all.append(feat)
'''# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert raw frequency counts into TF-IDF values
print('Performing tf-idf ...')
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun)
X = tfidf.fit_transform(feat_all)
print(type(X))
tfs_array = X.todense()
print(tfs_array.shape)'''


corpus = []
for feat in feat_all:
    corpus.append(" ".join(feat))

'''# Select a genre to see its word cloud
my_genre = 'Fantasy'
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

print('\n\nExtracting features ...')
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
feature_vector =  vectorizer.fit_transform(corpus).todense()
X = feature_vector
print('Feature vector size:\n\t',(np.array(feature_vector)).shape)
#print( vectorizer.vocabulary_ )

print('Creating label matrix.',end='\t')
y = [[] for _ in range(num_values)]
for i2 in range(num_values):
    for g2 in genre_unique:
        if (genres[i2].find(g2)!= -1):
            y[i2].append(1)
        else:
            y[i2].append(0)


from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
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

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(classifier, 'model.pkl')

with open("genres_files.txt", "w") as file:
    file.write(str(genre_unique))
#print(metrics.accuracy_score(y,predictions))
