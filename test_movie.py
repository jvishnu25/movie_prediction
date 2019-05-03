from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import joblib
import re
from sklearn.naive_bayes import GaussianNB
import numpy as np
vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('model.pkl')

def dummy_fun(doc):
    return doc


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



f=open("endgame.txt","r")

with open("genres_files.txt", "r") as file:
    my_genres = eval(file.readline())


    
contents =f.read()
contents = contents.strip()

data = text_preprocessor(contents)
data = " ".join(data)

X = vectorizer.transform([data])

X1 = X.todense()

y_pred= selector.predict(X1)
pred = y_pred.toarray()
result = list(np.where(pred == 1)[1])
print('\n\nPrediction:')
for r in result:
    print('\t*',my_genres[r])
