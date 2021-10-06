import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import numpy as np

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(sentence):
    result = [lemmatizer.lemmatize(x.lower()) for x in word_tokenize(sentence)]
    return result

def bag_of_words(sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for index, w in enumerate(words):
        if w in sentence: 
            bag[index] = 1
    return bag

