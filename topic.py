from gensim import corpora
from gensim import models
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

documents = [
    "Human machine interface"
    "A survey of user opinion on computer system response time"
    "The EPS user interface management system"
    "I like ice cream"
    "Students describe studying a subject they dislike"
    "High temperature and shark attacks"
    "Reading helps reduce stress"
    "People eating dinner at a fancy restaurant"
]

def preprocess(documents):
    #tokanize the text into words
    for doc in documents:
        stop_words = ["a", "of", "in", "to", "and", "for", "the",]
        punc =[".", ",","'", "(", ")"]

        good_words = []
        for word in words:  
            if  word.casefold() not in stop_words and ord not in punc:
                good_words.append(word)
        docs.append(good_words)
    print(docs)

    freq = defaultdict(int)
    for doc in docs:
        for word in doc:
            freq[word] +=1
    docs = [ [word for word in text if freq[word]>1]for text in docs]
    return docs

preprocess(preprocess(documents))
    #return reduced