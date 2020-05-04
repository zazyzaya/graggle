import pandas as pd
import nltk
import pickle
import os 
import numpy as np

from load_data import df
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

HOME = '/mnt/raid0_24TB/isaiah/repo/dash/data/'
MODEL_F = HOME + 'hv.model'
VECTORS_F = HOME + 'title_vecs.npz'

def pipeline(text):
    '''
    Processes raw text to remove all stopwords and lemmatize
    '''
    t = nltk.word_tokenize(text.lower())
    t = [LEMMATIZER.lemmatize(w) for w in t]
    t = [w for w in t if w not in STOPWORDS]
    t = [w for w in t if len(w) > 1]
    return ' '.join(t)

def build_vectors():
    text = []
    print("Cleaning data")
    for idx, row in df.iterrows():
        cleaned = pipeline(row['title'] + ' ' + row['abstract'])
        text.append(cleaned)
        
    print("Building vectors")
    hv = HashingVectorizer(n_features=2**10)
    hv.fit(text)
    X = hv.transform(text)

    print("Saving")
    save_npz(VECTORS_F, X)
    pickle.dump(hv, open(MODEL_F, 'wb+'))
    
    return X, hv

# Only build everything once if necessary 
if os.path.exists(VECTORS_F) and os.path.exists(MODEL_F):
    X = load_npz(VECTORS_F)
    hv = pickle.load(open(MODEL_F, 'rb'))
else:
    vectors, hv = build_vectors() 

# Return top 10 nearest neighbors to search term
def query(s, n=25):
    v = hv.transform([pipeline(s)])
    knn = cosine_similarity(X, v)
    
    # Return top 10 indexes 
    return knn[:,0].argsort()[-n:][::-1]
    