import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import random
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from collections import Counter
 

def save_pickle(name, classifier):
    byte_write = open(name, 'wb')
    pickle.dump(classifier, byte_write)
    byte_write.close()
    
def load_pickle(name):
    byte_read = open(name, 'rb')
    loaded = pickle.load(byte_read)
    byte_read.close()
    return loaded 
 

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v) 
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
     
 

short_pos = open("datasets/movie_review_positive.txt","r").read()
short_neg = open("datasets/movie_review_negative.txt","r").read()

# all_words = []
# documents = []

# #  j is adjectives, r is adverb, and v is verb
# #allowed_word_types = ["J","R","V"]
# allowed_word_types = ["J"]

# for r in short_pos.split('\n'):
#     documents.append((r, 'pos'))
#     words = word_tokenize(r)
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1][0] in allowed_word_types:
#             all_words.append(w[0].lower())
    
# for r in short_neg.split('\n'):
#     documents.append((r, 'neg'))
#     words = word_tokenize(r)
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1][0] in allowed_word_types:
#             all_words.append(w[0].lower())

# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)

# for w in short_pos_words:
#     all_words.append(w.lower())
    
# for w in short_neg_words:
#     all_words.append(w.lower())

all_words = load_pickle("word_frequencies/all_words.pickle")
documents = load_pickle("word_frequencies/documents.pickle")

 


# all_words = nltk.FreqDist(all_words)
# word_features = list(all_words.keys())[:10000]
word_features = load_pickle("word_frequencies/word_features.pickle")

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
        

# featuresets = [(find_features(rev), category) for (rev, category) in documents]
# random.shuffle(featuresets)
featuresets = load_pickle('word_frequencies/featureset.pickle') 

 

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:] 

# classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier = load_pickle('classifiers/NaiveBayesClassifier.pickle')
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
MNB_classifier = load_pickle("classifiers/MNB_classifier.pickle")
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
BernoulliNB_classifier = load_pickle("classifiers/BernoulliNB_classifier.pickle")
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
LogisticRegression_classifier = load_pickle("classifiers/LogisticRegression_classifier.pickle")
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
SGDClassifier_classifier = load_pickle("classifiers/SGDClassifier_classifier.pickle")
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100) 

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
LinearSVC_classifier = load_pickle("classifiers/LinearSVC_classifier.pickle")
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
NuSVC_classifier = load_pickle("classifiers/NuSVC_classifier.pickle")
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100) 

 

voted_classifier = VoteClassifier(MNB_classifier, 
                                  BernoulliNB_classifier, 
                                  LogisticRegression_classifier, 
                                  LinearSVC_classifier, 
                                  NuSVC_classifier)
print("Voted classifier Naive Bayes Algorithm accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

 

def sentiment(text):
    feats = find_features(text) 
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

 


# print(sentiment("I loved the movie. Probably going to buy and give it to my dad. He loved it")) # False negative
# print(sentiment("My dad loved it"))
# print(sentiment("The movie was awesome and I loved the acting"))
# print(sentiment("I hated that movie and I'll never watch that movie again"))

 

# print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[6][0]), "Confidence %: ", voted_classifier.confidence(testing_set[6][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[7][0]), "Confidence %: ", voted_classifier.confidence(testing_set[7][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[8][0]), "Confidence %: ", voted_classifier.confidence(testing_set[8][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[9][0]), "Confidence %: ", voted_classifier.confidence(testing_set[9][0])*100)

 

# save_pickle("classifiers/NaiveBayesClassifier.pickle", classifier)
# save_pickle("classifiers/MNB_classifier.pickle", MNB_classifier)
# save_pickle("classifiers/BernoulliNB_classifier.pickle", BernoulliNB_classifier)
# save_pickle("classifiers/LogisticRegression_classifier.pickle", LogisticRegression_classifier)
# save_pickle("classifiers/SGDClassifier_classifier.pickle", SGDClassifier_classifier)
# save_pickle("classifiers/LinearSVC_classifier.pickle", LinearSVC_classifier)
# save_pickle("classifiers/NuSVC_classifier.pickle", NuSVC_classifier)
 
# save_pickle("word_frequencies/all_word_frequencies.pickle", nltk.FreqDist(all_words))
# save_pickle("word_frequencies/all_words.pickle", all_words)
# save_pickle("word_frequencies/documents.pickle", documents)
# save_pickle("word_frequencies/featureset.pickle", featuresets)
# save_pickle("word_frequencies/word_features.pickle", word_features)

